"""Evaluate a trained checkpoint and write its predicted masks.

    python road_seg_test.py -c config/kaggle/ResNet34UNet.py -o /kaggle/working/results --rgb -t lr

Changes from upstream, all for Kaggle:

* runs on whatever device is available instead of hard-coding ``.cuda()``;
* writes each batch's masks as it goes, rather than holding every predicted mask
  in RAM until the end (2500 test tiles is several GB);
* logs the final metrics to MLflow, attaching them to the training run recorded
  in the weights directory so test numbers sit beside the training curves;
* clDice and APLS are computed here (they are too slow to run every epoch).
"""

import contextlib
import ttach as tta
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import time
from train_supervision import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools import mlflow_utils

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def label_to_rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [0, 0, 0]
    return mask_rgb


def img_writer(inp):
    (mask, mask_id, rgb) = inp
    if rgb:
        mask_name_tif = mask_id + '.png'
        mask_tif = label_to_rgb(mask)
        cv2.imwrite(mask_name_tif, mask_tif)
    else:
        mask_png = mask.astype(np.uint8)
        mask_name_png = mask_id + '.png'
        cv2.imwrite(mask_name_png, mask_png)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default='lr', choices=[None, "d4", "lr"])
    arg("--rgb", help="whether output rgb images", action='store_true')
    arg("--ckpt", type=Path, default=None,
        help="Checkpoint to evaluate (default: <weights_path>/<test_weights_name>.ckpt).")
    arg("--batch-size", type=int, default=2)
    arg("--num-workers", type=int, default=2)
    arg("--no-topology", action='store_true',
        help="Skip clDice/APLS (they skeletonise and graph every image).")
    arg("--no-mlflow", action='store_true', help="Do not log metrics to MLflow.")
    arg("--no-masks", action='store_true', help="Score only, do not write mask files.")
    return parser.parse_args()


def resolve_checkpoint(args, config) -> Path:
    if args.ckpt:
        return args.ckpt
    best = Path(config.weights_path) / f'{config.test_weights_name}.ckpt'
    if best.exists():
        return best
    last = Path(config.weights_path) / 'last.ckpt'
    if last.exists():
        print(f'[test] {best.name} not found, falling back to {last.name}')
        return last
    raise FileNotFoundError(f'no checkpoint in {config.weights_path}')


def log_to_mlflow(config, metrics: dict, extra_params: dict) -> None:
    """Attach the metrics to the training run if we can find it, else a new one."""
    try:
        import mlflow
    except ImportError:
        print('[test] mlflow not installed, skipping logging')
        return

    mlflow_utils.load_kaggle_secrets()
    run_id, saved_uri = mlflow_utils.saved_run_id(config.weights_path)
    tracking_uri = saved_uri or mlflow_utils.resolve_tracking_uri(
        config.get('mlflow_default_dir', 'mlruns'))
    mlflow.set_tracking_uri(tracking_uri)

    try:
        if run_id:
            mlflow.start_run(run_id=run_id)
        else:
            mlflow.set_experiment(config.get('experiment_name', config.weights_name))
            mlflow.start_run(run_name=f"{config.get('run_name', config.weights_name)}-test")
        mlflow.set_tags({'evaluated': 'true', **{f'test/{k}': v for k, v in extra_params.items()}})
        mlflow.log_metrics(metrics)
        print(f'[test] logged {len(metrics)} metrics to run {mlflow.active_run().info.run_id}')
    except Exception as exc:
        print(f'[test] could not log to MLflow: {exc}')
    finally:
        try:
            mlflow.end_run()
        except Exception:
            pass


def main():
    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = resolve_checkpoint(args, config)
    print(f'[test] loading {ckpt_path}')
    model = Supervision_Train.load_from_checkpoint(str(ckpt_path), config=config)
    model.to(device)
    model.eval()

    evaluator = Evaluator(num_class=config.num_classes, topology=not args.no_topology)
    evaluator.reset()
    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[0, 90, 180, 270])
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    test_dataset = config.test_dataset

    # Score in the precision the model was trained and validated in: on a T4,
    # fp16 runs on the tensor cores and fp32 does not.
    amp_dtype = {'16-mixed': torch.float16, 'bf16-mixed': torch.bfloat16}.get(str(config.get('precision')))
    autocast = (torch.autocast('cuda', dtype=amp_dtype) if amp_dtype and device.type == 'cuda'
                else contextlib.nullcontext())
    torch.backends.cudnn.benchmark = not config.get('deterministic', False)

    write_pool = ThreadPool(processes=max(2, mp.cpu_count()))
    pending = []
    write_time = 0.0

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        for input in tqdm(test_loader):
            with autocast:
                raw_predictions = model(input['img'].to(device))
            image_ids = input["img_id"]
            masks_true = input.get('gt_semantic_seg')

            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1).cpu().numpy().astype(np.uint8)

            batch = []
            for i in range(predictions.shape[0]):
                mask = predictions[i]
                if masks_true is not None:
                    evaluator.add_batch(pre_image=mask, gt_image=masks_true[i].cpu().numpy())
                if not args.no_masks:
                    batch.append((mask, str(args.output_path / image_ids[i]), args.rgb))

            # Write as we go; accumulating every mask first would cost GBs of RAM.
            if batch:
                t0 = time.time()
                pending.append(write_pool.map_async(img_writer, batch))
                write_time += time.time() - t0

    t0 = time.time()
    for result in pending:
        result.wait()
    write_pool.close()
    write_pool.join()
    write_time += time.time() - t0
    if not args.no_masks:
        print('images writing spends: {} s'.format(write_time))

    iou_per_class = evaluator.Intersection_over_Union()
    f1_per_class = evaluator.F1()
    apls_score = evaluator.APLS()
    OA = evaluator.OA()
    precision = evaluator.Precision()
    recall = evaluator.Recall()
    clDice = evaluator.clDice()
    for class_name, class_iou, class_f1 in zip(config.CLASSES, iou_per_class, f1_per_class):
        print('F1_{}:{:.4f}, IOU_{}:{:.4f}'.format(class_name, class_f1, class_name, class_iou))
    print('F1:{:.4f}, mIOU:{:.4f}, OA:{:.4f}, P:{:.4f}, R:{:.4f}, clDice:{:.4f}, APLS:{:.4f}'.format(np.nanmean(f1_per_class[:-1]), np.nanmean(iou_per_class[:-1]), OA,
                                                    np.nanmean(precision[:-1]), np.nanmean(recall[:-1]), clDice, apls_score))

    if not args.no_mlflow:
        # f1_per_class[:-1] drops Background, so these are the road-only figures
        # that the paper reports; the per-class values are logged too.
        metrics = {
            'test_F1': float(np.nanmean(f1_per_class[:-1])),
            'test_mIoU_foreground': float(np.nanmean(iou_per_class[:-1])),
            'test_mIoU': float(np.nanmean(iou_per_class)),
            'test_OA': float(OA),
            'test_Precision': float(np.nanmean(precision[:-1])),
            'test_Recall': float(np.nanmean(recall[:-1])),
        }
        for class_name, iou, f1, p, r in zip(config.CLASSES, iou_per_class, f1_per_class,
                                             precision, recall):
            metrics[f'test_IoU_{class_name}'] = float(iou)
            metrics[f'test_F1_{class_name}'] = float(f1)
            metrics[f'test_Precision_{class_name}'] = float(p)
            metrics[f'test_Recall_{class_name}'] = float(r)
        if not args.no_topology:
            metrics['test_clDice'] = float(clDice)
            metrics['test_APLS'] = float(apls_score)
        log_to_mlflow(config, metrics, {
            'checkpoint': ckpt_path.name,
            'tta': str(args.tta),
        })


if __name__ == "__main__":
    main()
