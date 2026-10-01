"""Supervised training for the road segmentation configs.

GeoSeg's Lightning training loop, logging to MLflow instead of a CSV file, with
the knobs a Kaggle session needs: mixed precision, channels_last, a wall-clock
stop, automatic resume, and DDP-correct epoch metrics accumulated on the GPU.

    python train_supervision.py -c config/kaggle/ResNet34UNet.py
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Timer, TQDMProgressBar
from tools.cfg import py2cfg
import os
import torch
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from tools import mlflow_utils
from tools.callbacks import EpochTimer, MLflowPredictionImages
import random
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

torch.set_float32_matmul_precision('high')

# Datasets whose last class is a "don't score it" background: upstream averages
# over classes[:-1] for these, i.e. it reports the road class alone.
FOREGROUND_ONLY_DATASETS = ('vaihingen', 'potsdam', 'whu', 'mass', 'inria')


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def resolve_channels_last(setting) -> bool:
    """'auto' means NHWC on tensor-core GPUs (compute >= 7.0) and NCHW elsewhere."""
    setting = str(setting).strip().lower()
    if setting in ('0', 'false', 'no', 'off') or not torch.cuda.is_available():
        return False
    if setting == 'auto':
        return torch.cuda.get_device_capability(0)[0] >= 7
    return True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.num_classes = config.num_classes

        self.loss = config.loss

        self.channels_last = resolve_channels_last(config.get('channels_last', 'auto'))
        if self.channels_last:
            self.net = self.net.to(memory_format=torch.channels_last)

        # Confusion matrices live on the GPU and reach the host once per epoch;
        # copying every predicted mask to numpy stalled each training step.
        nc = self.num_classes
        self.register_buffer('cm_train', torch.zeros(nc, nc, dtype=torch.long), persistent=False)
        self.register_buffer('cm_val', torch.zeros(nc, nc, dtype=torch.long), persistent=False)

        # Topology metrics (clDice / APLS) skeletonise every image on the CPU and
        # build two graphs per image. That is affordable at test time, not inside
        # a training loop, so each split gets its own switch.
        self.metrics = {
            'train': Evaluator(num_class=nc, topology=bool(config.get('train_topology_metrics', False))),
            'val': Evaluator(num_class=nc, topology=bool(config.get('val_topology_metrics', False))),
        }

    def forward(self, x):
        # only net is used in the prediction/inference
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        # Hand back NCHW logits: DiceLoss .view()s them, which an NHWC tensor refuses.
        return self.net(x).contiguous()

    def training_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']

        prediction = self(img)
        loss = self.loss(prediction, mask)
        self._update_metrics('train', prediction, mask)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    # ------------------------------------------------------------------ #
    # metric aggregation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _update_metrics(self, stage, prediction, mask):
        pred = prediction.argmax(dim=1)
        cm = getattr(self, f'cm_{stage}')
        # One masked count per cell: no bincount, so no host sync per batch.
        for true_class in range(self.num_classes):
            is_true = mask == true_class
            for pred_class in range(self.num_classes):
                cm[true_class, pred_class] += (is_true & (pred == pred_class)).sum()

        evaluator = self.metrics[stage]
        if evaluator.topology:
            for gt_i, pred_i in zip(mask.cpu().numpy(), pred.cpu().numpy()):
                evaluator.add_batch(gt_i, pred_i)

    def _merge_across_ranks(self, stage):
        """Confusion matrix and topology accumulators, summed over DDP ranks.

        Without this each rank reports metrics over its own shard, which makes
        multi-GPU numbers quietly disagree with single-GPU ones.
        """
        world_size = self.trainer.world_size if self.trainer is not None else 1
        evaluator = self.metrics[stage]

        cm = getattr(self, f'cm_{stage}').to(torch.float64)
        extras = torch.tensor([
            evaluator.cldice_tprec_num, evaluator.cldice_tprec_den,
            evaluator.cldice_trec_num, evaluator.cldice_trec_den,
            float(np.sum(evaluator.apls_scores)), float(len(evaluator.apls_scores)),
        ], device=self.device, dtype=torch.float64)

        if world_size > 1:
            cm = self.all_gather(cm).sum(dim=0)
            extras = self.all_gather(extras).sum(dim=0)

        merged = Evaluator(num_class=evaluator.num_class, topology=evaluator.topology)
        merged.add_confusion_matrix(cm.cpu().numpy())
        extras = extras.cpu().numpy()
        merged.cldice_tprec_num, merged.cldice_tprec_den = extras[0], extras[1]
        merged.cldice_trec_num, merged.cldice_trec_den = extras[2], extras[3]
        if extras[5] > 0:
            # APLS is a mean over images; replay it as a single averaged sample.
            merged.apls_scores = [extras[4] / extras[5]]
        return merged

    def _epoch_metrics(self, stage):
        """Reproduces upstream's per-dataset averaging, plus per-class values."""
        merged = self._merge_across_ranks(stage)
        iou_per_class = merged.Intersection_over_Union()
        f1_per_class = merged.F1()

        log_name = str(self.config.log_name)
        if any(tag in log_name for tag in FOREGROUND_ONLY_DATASETS):
            mIoU = np.nanmean(iou_per_class[:-1])
            F1 = np.nanmean(f1_per_class[:-1])
        else:
            mIoU = np.nanmean(iou_per_class)
            F1 = np.nanmean(f1_per_class)
        OA = np.nanmean(merged.OA())

        eval_value = {'mIoU': float(np.round(mIoU, 6)),
                      'F1': float(np.round(F1, 6)),
                      'OA': float(np.round(OA, 6))}
        self.print(f'{stage}:', eval_value)

        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = np.round(iou, 6)
        self.print(iou_value)

        log_dict = {f'{stage}_mIoU': mIoU, f'{stage}_F1': F1, f'{stage}_OA': OA}
        # Per-class metrics matter here: with two classes, mIoU is dominated by
        # background, so the road column is the number to read.
        for class_name, iou, f1 in zip(self.config.classes, iou_per_class, f1_per_class):
            log_dict[f'{stage}_IoU_{class_name}'] = float(iou)
            log_dict[f'{stage}_F1_{class_name}'] = float(f1)
        if merged.topology:
            cldice, apls = merged.clDice(), merged.APLS()
            self.print(f'{stage} topology: clDice={cldice:.6f} APLS={apls:.6f}')
            log_dict[f'{stage}_clDice'] = float(cldice)
            log_dict[f'{stage}_APLS'] = float(apls)

        getattr(self, f'cm_{stage}').zero_()
        self.metrics[stage].reset()
        return log_dict

    def on_train_epoch_end(self):
        log_dict = self._epoch_metrics('train')
        # Values are already reduced across ranks, so every rank logs the same
        # numbers -- ModelCheckpoint under DDP needs the monitored key present.
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        prediction = self(img)
        self._update_metrics('val', prediction, mask)

        loss_val = self.loss(prediction, mask)
        self.log('val_loss', loss_val, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss_val": loss_val}

    def on_validation_epoch_end(self):
        self.print(" ")
        log_dict = self._epoch_metrics('val')
        self.print("======================")
        # Values are already reduced across ranks, so every rank logs the same
        # numbers -- ModelCheckpoint under DDP needs the monitored key present.
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler
        if callable(lr_scheduler):
            # A factory: OneCycleLR needs the run's total optimiser steps, which
            # depend on the GPU count, batch size and limit_train_batches.
            lr_scheduler = lr_scheduler(optimizer, int(self.trainer.estimated_stepping_batches))

        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': lr_scheduler,
                                 'interval': self.config.get('lr_scheduler_interval', 'epoch')}}

    def train_dataloader(self):

        return self.config.train_loader

    def val_dataloader(self):

        return self.config.val_loader


def resolve_resume_path(config):
    """Explicit resume path, else the last checkpoint this output dir holds."""
    if config.resume_ckpt_path:
        return str(config.resume_ckpt_path)
    if not config.get('auto_resume', True):
        return None
    last = Path(config.weights_path) / 'last.ckpt'
    if last.exists():
        print(f'[train] resuming from {last}')
        return str(last)
    return None


# training
def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(config.get('seed', 42), config.get('deterministic', False))

    Path(config.weights_path).mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k,
                                        monitor=config.monitor,
                                        save_last=config.save_last,
                                        mode=config.monitor_mode,
                                        dirpath=config.weights_path,
                                        filename=config.weights_name)
    callbacks = [checkpoint_callback, LearningRateMonitor(logging_interval='epoch'), EpochTimer()]

    refresh_rate = config.get('progress_bar_refresh_rate', 1)
    if refresh_rate > 0:
        callbacks.append(TQDMProgressBar(refresh_rate=refresh_rate))

    # Kaggle kills the session on a hard wall clock. Stopping a little early
    # leaves time for the final checkpoint write and the MLflow flush, and
    # `last.ckpt` lets the next session pick up where this one stopped.
    max_time = config.get('max_time', None)
    if max_time:
        callbacks.append(Timer(duration=max_time, interval='step'))

    if config.get('log_prediction_images', True):
        callbacks.append(MLflowPredictionImages(
            num_samples=config.get('prediction_image_samples', 4),
            every_n_epochs=config.get('prediction_image_every_n_epochs', 5),
        ))

    model_name = config.get('model_name', config.weights_name)
    logger = mlflow_utils.build_logger(
        experiment_name=config.get('experiment_name', model_name),
        run_name=config.get('run_name', config.weights_name),
        default_store_dir=config.get('mlflow_default_dir', 'mlruns'),
        state_dir=config.weights_path,
        tags={'dataset': config.get('dataset_name', 'unknown'), 'model': model_name},
        resume=config.get('auto_resume', True),
        log_model=config.get('mlflow_log_model', False),
    )
    if config.get('mlflow_system_metrics', True):
        mlflow_utils.enable_system_metrics()

    model = Supervision_Train(config)
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

    mlflow_utils.log_hyperparams(logger, {
        **mlflow_utils.config_params(config),
        **mlflow_utils.env_params(),
        **mlflow_utils.model_params(model.net),
        'env/channels_last': model.channels_last,
    })
    mlflow_utils.log_artifact(logger, args.config_path, 'config')

    trainer = pl.Trainer(devices=config.gpus,
                        max_epochs=config.max_epoch,
                        accelerator='auto',
                        check_val_every_n_epoch=config.check_val_every_n_epoch,
                        callbacks=callbacks,
                        strategy=config.get('strategy', 'auto'),
                        precision=config.get('precision', '32-true'),
                        accumulate_grad_batches=config.get('accumulate_grad_batches', 1),
                        gradient_clip_val=config.get('gradient_clip_val', None),
                        log_every_n_steps=config.get('log_every_n_steps', 50),
                        num_sanity_val_steps=config.get('num_sanity_val_steps', 2),
                        limit_train_batches=config.get('limit_train_batches', 1.0),
                        limit_val_batches=config.get('limit_val_batches', 1.0),
                        enable_progress_bar=refresh_rate > 0,
                        logger=logger)

    status = 'FINISHED'
    try:
        trainer.fit(model=model, ckpt_path=resolve_resume_path(config))
    except BaseException:
        status = 'FAILED'
        raise
    finally:
        summary = {
            'best_model_path': checkpoint_callback.best_model_path,
            'best_model_score': (float(checkpoint_callback.best_model_score)
                                 if checkpoint_callback.best_model_score is not None else None),
            'monitor': config.monitor,
            'epochs_completed': trainer.current_epoch,
        }
        mlflow_utils.log_dict(logger, summary, 'training_summary.json')
        stopped_early = status == 'FINISHED' and trainer.current_epoch < config.max_epoch
        if stopped_early:
            # The wall-clock Timer stopped us. Reopen the MLflow run (Lightning
            # has just closed it) so the next Kaggle session resumes into it
            # instead of opening a second one.
            status = 'RUNNING (resumable)'
            mlflow_utils.set_resumable(logger)
            print(f'[train] stopped at epoch {trainer.current_epoch}/{config.max_epoch}; '
                  f'rerun to continue from last.ckpt')
        else:
            mlflow_utils.set_terminated(logger, status)
        print(f'[train] {status}; best {config.monitor}='
              f'{summary["best_model_score"]} at {summary["best_model_path"]}')


if __name__ == "__main__":
    main()
