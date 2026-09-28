# RFM-UNet road extraction on Kaggle, tracked with MLflow

[RFM-UNet](https://github.com/FF7CA/RFMUNet) — *Hybrid Frequency-Mamba UNet for Remote
Sensing Road Extraction* — trained on the [DeepGlobe Road Extraction
dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset) in a
Kaggle GPU session, with every run recorded in MLflow.

The model is a VMamba-tiny (ADAMamba) encoder, MAFM multi-scale skip fusion, and a
frequency/spatial dual-branch decoder. **The model maths is upstream's and untouched.**
This repo changes where it runs and how it reports.

Start with [`rfmunet-kaggle.ipynb`](rfmunet-kaggle.ipynb).

---

## Read this first: the CUDA kernel

Upstream's requirements say NVIDIA compute capability ≥ 8.0 (Ampere) and ≥ 24 GB VRAM.
Kaggle gives you a Turing T4 (compute 7.5, 16 GB) or a Pascal P100 (6.0). Those numbers
are not about the model — they come from the *prebuilt* `mamba-ssm` wheels — but one part
of the requirement is real:

**RFM-UNet needs a compiled selective-scan CUDA kernel.** The VMamba code ships a
pure-PyTorch fallback that loops over every timestep in Python. At 512×512 the first
encoder stage has a sequence length of 16384, so a single SSM block costs ~16k Python
iterations and materialises tensors of shape `(B, 768, 16, 16384)`. It is numerically
correct — the notebook checks it against the kernel — and useful for a shape test, but it
is not within two orders of magnitude of trainable, and it will not fit in 16 GB.

Either kernel satisfies it, and [`geoseg/models/ADAMamba.py`](geoseg/models/ADAMamba.py)
picks up whichever is importable:

| module | install | notes |
| --- | --- | --- |
| `selective_scan_cuda` | `pip install mamba-ssm==2.2.4` | fast if a wheel matches your torch/CUDA/arch; a source build needs `nvcc` |
| `selective_scan_cuda_oflex` | VMamba's `kernels/selective_scan` | compiles for whatever `TORCH_CUDA_ARCH_LIST` you give it, so it is the dependable route on a T4 |

Step 3 of the notebook tries the quick route, builds from source if needed (10–25 min),
then **verifies the kernel against the PyTorch reference** — an import succeeding does not
prove the cubin matches your GPU. Build it once, save the wheel as a Kaggle Dataset, attach
it to later sessions.

`SELECTIVE_SCAN_BACKEND=auto|mamba|oflex|torch` forces the choice. Every run logs which one
it used as the `env/selective_scan` param, so a mysteriously slow run is one glance away
from an explanation.

---

## Quick start

On Kaggle:

1. Accelerator **GPU** (T4 x2 or P100), **Internet on**.
2. Attach `balraj98/deepglobe-road-extraction-dataset`.
3. Run [`rfmunet-kaggle.ipynb`](rfmunet-kaggle.ipynb) top to bottom. It probes the GPU,
   installs deps, sorts out the kernel, prepares the data, runs a 2-batch smoke test, then
   trains and evaluates.

Anywhere else:

```bash
pip install -r requirements-kaggle.txt          # plus torch, and a selective-scan kernel
python tools/prepare_deepglobe.py --src /path/to/deepglobe --dst ./data/deepglobe
DEEPGLOBE_ROOT=./data/deepglobe OUT_ROOT=. \
  python train_supervision.py -c config/kaggle/RFMUNet.py
DEEPGLOBE_ROOT=./data/deepglobe OUT_ROOT=. \
  python road_seg_test.py -c config/kaggle/RFMUNet.py -o ./results --rgb -t lr
```

---

## Data preparation

`tools/prepare_deepglobe.py` handles two things that are easy to get wrong.

**Only `train/` is labelled.** DeepGlobe's `valid/` and `test/` directories are the
unlabelled competition holdouts. The 6226 labelled pairs in `train/` are split 80/10/10
into train/val/test. The split is on *source image id, before tiling*, so crops of one
scene never end up in two different splits.

**Masks must be class indices, not 0/255.** `geoseg/datasets/dpgb.py` feeds the mask
straight to the loss, and both `dpgb.PALETTE` and `tools/metric.py` treat **class 0 as
road** (white first in the palette; `_skeletonize` uses `mask == 0`). So the script writes
road → 0, background → 1. Handing the raw 0/255 mask to training instead would be silently
catastrophic: `EdgeLoss(ignore_index=255)` would discard every road pixel and the model
would happily converge to all-background with a fine-looking OA. The script verifies the
encoding after writing rather than trusting it.

Source images are 1024², the model trains at 512². `--mode split` (default) cuts four
non-overlapping tiles per image, keeping thin roads at native resolution;
`--mode resize` downscales instead — 4× less data and 4× faster epochs, at the cost of
one-pixel roads. `--fraction 0.25` subsamples for a quick pass.

---

## What gets logged

Local file store by default (`/kaggle/working/mlruns`, kept as notebook output; read it
with `mlflow ui --backend-store-uri ./mlruns`). Set a `MLFLOW_TRACKING_URI` **env var or
Kaggle Secret** to use a remote server instead — no code change, and credentials come from
the same two places (`MLFLOW_TRACKING_USERNAME` / `_PASSWORD` / `_TOKEN`).

Per epoch, for train and val:

- `*_mIoU`, `*_F1`, `*_OA`, and `train_loss` / `val_loss`
- `*_IoU_Road`, `*_F1_Road`, `*_IoU_Background`, `*_F1_Background`

With two classes, mIoU is dominated by background, so **`val_IoU_Road` is the number to
watch.** `mIoU` is kept as-is for comparability with the paper's DeepGlobe numbers — see
"Deviations" below for exactly what upstream averages.

At test time (`road_seg_test.py`), on the same run: `test_IoU_*`, `test_F1_*`,
`test_Precision_*`, `test_Recall_*`, plus the topology metrics **`test_clDice`** and
**`test_APLS`**. Those two skeletonise and graph every image on the CPU, which is why they
are off during training (`VAL_TOPOLOGY_METRICS=1` to enable anyway).

Also logged: every scalar config value, `env/*` (GPU, capability, torch, CUDA, selective-scan
backend, git sha, upstream sha), `model/params_*`, CPU/GPU/RAM system metrics, the config
file as an artifact, a `training_summary.json`, and a prediction preview grid every 5 epochs
under `val_predictions/` — input, ground truth, prediction, and an error map (green hit,
red false positive, blue missed road).

**Resume behaviour.** The run id is cached in `<weights_path>/mlflow_run.json`. A session
killed by the wall clock leaves the run open; rerunning picks up `last.ckpt` *and* the same
MLflow run, so a 105-epoch schedule spread over several Kaggle sessions is one continuous
set of curves.

---

## Configuration

[`config/kaggle/RFMUNet.py`](config/kaggle/RFMUNet.py) reads an env var for every knob, so
the notebook retunes a run without editing files:

```bash
TRAIN_BATCH_SIZE=4 MAX_EPOCH=45 GPUS=2 STRATEGY=ddp \
  python train_supervision.py -c config/kaggle/RFMUNet.py
```

| variable | default | |
| --- | --- | --- |
| `DEEPGLOBE_ROOT` | `/kaggle/working/data/deepglobe` | prepared split (an attached Dataset works) |
| `OUT_ROOT` | `/kaggle/working` | checkpoints, `mlruns`, results |
| `TRAIN_BATCH_SIZE` / `ACCUM_GRAD_BATCHES` | `2` / `6` | effective batch 12, as upstream |
| `PRECISION` | `16-mixed` | `32-true` to match upstream exactly |
| `USE_GRAD_CHECKPOINT` | `1` | encoder activation checkpointing |
| `MAX_EPOCH` | `105` | cosine warm restarts at 15, 45, 105 |
| `MAX_TIME` | `00:08:00:00` | `DD:HH:MM:SS` wall-clock stop; `''` disables |
| `GPUS` / `STRATEGY` | `1` / `auto` | `2` / `ddp` for T4 x2 |
| `MONITOR` | `val_mIoU` | `val_IoU_Road` to select checkpoints on road IoU |
| `PRETRAINED_ENCODER` | `0` | VMamba-tiny ImageNet weights, loaded non-strictly |
| `LIMIT_TRAIN_BATCHES` / `LIMIT_VAL_BATCHES` | `1.0` | fractions turn the config into a smoke test |
| `GRADIENT_CLIP_VAL` | `0` (off) | set `1.0` if fp16 sends the loss to NaN |
| `SELECTIVE_SCAN_BACKEND` | `auto` | `mamba` / `oflex` / `torch` |

`config/DPGB`, `config/Mass` and `config/CHN6` are upstream's originals, kept unmodified
for reference; they point at `/root/autodl-tmp` paths.

---

## Deviations from upstream

Vendored from [FF7CA/RFMUNet](https://github.com/FF7CA/RFMUNet) at
`bd04e15` (see `.rfmunet-upstream-sha`). The model, loss, optimiser, LR schedule and
augmentations are unchanged. Everything else:

**Made reachable / portable**

- `vanilla_vmamba_tiny(**kwargs)` accepted `**kwargs` and discarded them, so
  `use_checkpoint` could not be set from a config. They are now forwarded to `VSSM`.
- `selective_scan_cuda_oflex` was called but never imported — the `"oflex"` branch was
  dead code. Both kernels are now imported and `selective_scan_backend()` picks one, rather
  than dropping to the Python-loop fallback whenever `mamba-ssm` specifically was absent.
- VMamba's gradient checkpointing called `checkpoint.checkpoint()` without
  `use_reentrant`, so it took torch's reentrant default. That re-enters the autograd
  engine during backward and fires DDP's gradient hooks twice per parameter, killing
  any multi-GPU run on its first optimiser step with *"marked as ready twice"*. Now
  `use_reentrant=False`, which is gradient-identical and DDP-safe.
- `fvcore` is imported lazily; it is only used by `VSSM.flops()`, which training never calls.
- `EdgeLoss`'s Laplacian kernel used `.cuda()`; now `.to(x.device)`, so CPU runs work.
- `dpgb.path` reads `DEEPGLOBE_ROOT` instead of hard-coding `/root/autodl-tmp/data/DeepGlobe`.

**Training loop**

- `CSVLogger` → `MLFlowLogger`, plus the params/artifacts/previews listed above.
- Epoch metrics are reduced across DDP ranks. Upstream's per-rank `Evaluator` means a
  2-GPU run reports metrics over half the data per rank, which does not match a 1-GPU run.
- `Evaluator(topology=False)` skips clDice/APLS. Upstream computes them inside
  `add_batch`, i.e. a skeletonisation and two graph builds per image per epoch, on the
  training loop's process. They are on for `road_seg_test.py` and off during training.
- Added: mixed precision, gradient accumulation, a wall-clock `Timer`, `LearningRateMonitor`,
  automatic resume from `last.ckpt`, and `limit_*_batches` for smoke tests.
- `road_seg_test.py` writes masks per batch instead of accumulating every predicted mask
  in memory (a few GB over the test split), and runs on CPU if there is no GPU.

**Metric averaging, unchanged but worth knowing.** `train_supervision.py` keys its
averaging off substrings of `log_name`. `'mass'` does not match upstream's
`.../Mass/.../Mass_log` (case-sensitive), and `'dpgb'` is not in the list at all, so in
practice *both* upstream configs average IoU/F1 over road **and** background. This repo
reproduces that so the numbers stay comparable, and logs `*_IoU_Road` separately so the
road-only figure is always available.

---

## Layout

```
rfmunet-kaggle.ipynb        Kaggle runner: probe -> deps -> kernel -> data -> smoke -> train -> eval
train_supervision.py        training entry point (MLflow)
road_seg_test.py            evaluation + mask export (MLflow)
config/kaggle/RFMUNet.py    DeepGlobe on a Kaggle GPU, env-var driven
config/{DPGB,Mass,CHN6}/    upstream configs, unmodified
geoseg/models/              RFMUNet, ADAMamba (VMamba), MAFM, DualSpec
geoseg/losses/              EdgeLoss and the rest of the GeoSeg loss zoo
geoseg/datasets/            dpgb (DeepGlobe), mass, CHN6
tools/prepare_deepglobe.py  Kaggle dataset -> 512 tiles, masks re-encoded, verified
tools/mlflow_utils.py       tracking URI resolution, secrets, params, run-id resume
tools/callbacks.py          validation prediction previews
tools/{cfg,metric,utils}.py upstream helpers (metric.py: topology switch + DDP merge)
```

## Credit

RFM-UNet by the authors of *RFM-UNet: Hybrid Frequency-Mamba UNet for Remote Sensing Road
Extraction* ([FF7CA/RFMUNet](https://github.com/FF7CA/RFMUNet), MIT). Training scaffolding
from [GeoSeg](https://github.com/WangLibo1995/GeoSeg); encoder from
[VMamba](https://github.com/MzeroMiko/VMamba).
