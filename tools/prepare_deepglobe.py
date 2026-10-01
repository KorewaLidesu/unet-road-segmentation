"""Turn the Kaggle DeepGlobe Road Extraction dataset into the layout training wants.

    python tools/prepare_deepglobe.py \
        --src /kaggle/input/deepglobe-road-extraction-dataset \
        --dst /kaggle/working/data/deepglobe

Two things have to happen and both are easy to get wrong:

1. **Only ``train/`` is usable.** The dataset's ``valid/`` and ``test/`` folders
   are the unlabelled competition holdouts, so the 6226 labelled pairs in
   ``train/`` are split here into train/val/test. The split is on *source image*
   id, before tiling, so no two crops of the same scene land in different splits.

2. **Masks must be class indices, not 0/255.** ``geoseg.datasets.dpgb`` feeds the
   mask straight to the loss, and ``tools.metric`` treats class 0 as road (see
   ``PALETTE``: white first). So road -> 0, background -> 1. Handing the raw
   0/255 mask to training instead would be silently catastrophic: every road
   pixel would be read as ``ignore_index=255``.

Source images are 1024x1024. Each becomes four non-overlapping 512 tiles by
default (``--mode split``), preserving the native resolution of thin roads:
validation and testing score every tile, and training cuts its random crops
out of them. ``--mode resize`` halves each image to one 512 tile instead: 4x
less to validate and test, at the cost of one-pixel roads.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

ROAD, BACKGROUND = 0, 1
TILE = 512
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# discovery and splitting
# --------------------------------------------------------------------------- #

def find_pairs(src: Path) -> list[tuple[Path, Path]]:
    """Every ``*_sat.jpg`` under ``src`` that has a matching ``*_mask.png``."""
    pairs = []
    for sat in sorted(src.rglob("*_sat.jpg")):
        mask = sat.with_name(sat.name.replace("_sat.jpg", "_mask.png"))
        if mask.exists():
            pairs.append((sat, mask))
    return pairs


def split_pairs(pairs, ratios=(0.8, 0.1, 0.1), seed=42, fraction=1.0):
    """Deterministic split on source id, so tiles never leak across splits."""
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    if fraction < 1.0:
        pairs = pairs[: max(1, int(round(len(pairs) * fraction)))]

    n = len(pairs)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    return {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }


# --------------------------------------------------------------------------- #
# conversion
# --------------------------------------------------------------------------- #

def encode_mask(mask: np.ndarray) -> np.ndarray:
    """0/255 (or RGB) road mask -> class indices, road=0 background=1."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return np.where(mask >= 128, ROAD, BACKGROUND).astype(np.uint8)


def _tiles(img: np.ndarray, mask: np.ndarray, mode: str, tile: int):
    """Yield ``(suffix, image, mask)`` crops of size ``tile``."""
    if mode == "resize":
        yield ("", cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA),
               cv2.resize(mask, (tile, tile), interpolation=cv2.INTER_NEAREST))
        return

    h, w = mask.shape[:2]
    if h < tile or w < tile:  # pad short edges rather than skip the image
        pad_h, pad_w = max(0, tile - h), max(0, tile - w)
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT,
                                  value=int(BACKGROUND))
        h, w = mask.shape[:2]

    for r in range(0, h - tile + 1, tile):
        for c in range(0, w - tile + 1, tile):
            yield (f"_r{r // tile}c{c // tile}",
                   img[r:r + tile, c:c + tile], mask[r:r + tile, c:c + tile])


def convert_one(job):
    """Worker: one source pair -> N tile files. Returns (written, road_px, total_px)."""
    sat_path, mask_path, img_dir, mask_dir, mode, tile, quality = job
    img = cv2.imread(str(sat_path), cv2.IMREAD_COLOR)
    raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if img is None or raw is None:
        return 0, 0, 0, f"unreadable: {sat_path.name}"

    mask = encode_mask(raw)
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    stem = sat_path.name[: -len("_sat.jpg")]
    written, road_px, total_px = 0, 0, 0
    for suffix, img_tile, mask_tile in _tiles(img, mask, mode, tile):
        name = f"{stem}{suffix}"
        cv2.imwrite(str(Path(img_dir) / f"{name}.jpg"), img_tile,
                    [cv2.IMWRITE_JPEG_QUALITY, quality])
        cv2.imwrite(str(Path(mask_dir) / f"{name}.png"), mask_tile)
        written += 1
        road_px += int((mask_tile == ROAD).sum())
        total_px += mask_tile.size
    return written, road_px, total_px, None


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def manifest_for(args, pair_count: int) -> dict:
    return {
        "src": str(args.src),
        "mode": args.mode,
        "tile": args.tile,
        "seed": args.seed,
        "fraction": args.fraction,
        "ratios": [args.train_ratio, args.val_ratio],
        "jpeg_quality": args.quality,
        "source_pairs": pair_count,
        "road_class_index": ROAD,
    }


def verify(dst: Path, num_classes: int = 2, samples: int = 40) -> None:
    """Spot-check that masks really are class indices and shapes line up."""
    rng = random.Random(0)
    for split in SPLITS:
        masks = sorted((dst / f"{split}_masks").glob("*.png"))
        images = dst / f"{split}_images"
        assert masks, f"no masks in {split}_masks"
        for mask_path in rng.sample(masks, min(samples, len(masks))):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            assert mask is not None, f"unreadable {mask_path}"
            values = np.unique(mask)
            assert values.max() < num_classes, (
                f"{mask_path} holds values {values.tolist()}; expected only "
                f"0..{num_classes - 1}. Raw 0/255 DeepGlobe masks would train to garbage."
            )
            img_path = images / f"{mask_path.stem}.jpg"
            assert img_path.exists(), f"mask without image: {mask_path}"
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            assert img.shape[:2] == mask.shape[:2], (
                f"shape mismatch for {mask_path.stem}: {img.shape[:2]} vs {mask.shape[:2]}")
    print("[prep] verification passed")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path,
                        default=Path(os.environ.get(
                            "RAW_DEEPGLOBE_ROOT",
                            "/kaggle/input/deepglobe-road-extraction-dataset")),
                        help="Kaggle DeepGlobe dataset root (must contain train/*_sat.jpg).")
    parser.add_argument("--dst", type=Path,
                        default=Path(os.environ.get("DEEPGLOBE_ROOT",
                                                    "/kaggle/working/data/deepglobe")),
                        help="Where to write the prepared split.")
    parser.add_argument("--mode", choices=("split", "resize"), default="split",
                        help="split: four 512 tiles per 1024 image (default). "
                             "resize: one 512 downscale, 4x smaller and faster.")
    parser.add_argument("--tile", type=int, default=TILE)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Use only this fraction of source images (quick runs).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=95, help="Output JPEG quality.")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true",
                        help="Re-prepare even if the destination already matches.")
    args = parser.parse_args(argv)

    if not args.src.exists():
        print(f"[prep] source not found: {args.src}\n"
              f"       On Kaggle, add the 'DeepGlobe Road Extraction Dataset' to the "
              f"notebook and check the path under /kaggle/input.", file=sys.stderr)
        return 2

    pairs = find_pairs(args.src)
    if not pairs:
        print(f"[prep] no *_sat.jpg / *_mask.png pairs under {args.src}", file=sys.stderr)
        return 2
    print(f"[prep] found {len(pairs)} labelled pairs under {args.src}")

    args.dst.mkdir(parents=True, exist_ok=True)
    manifest_path = args.dst / ".prepared.json"
    wanted = manifest_for(args, len(pairs))
    if manifest_path.exists() and not args.force:
        try:
            existing = json.loads(manifest_path.read_text())
        except Exception:
            existing = {}
        if {k: existing.get(k) for k in wanted} == wanted:
            counts = existing.get("counts", {})
            print(f"[prep] already prepared at {args.dst}: {counts} (use --force to redo)")
            return 0
        print("[prep] existing preparation used different settings; redoing")

    splits = split_pairs(pairs, (args.train_ratio, args.val_ratio,
                                 1.0 - args.train_ratio - args.val_ratio),
                         seed=args.seed, fraction=args.fraction)

    counts, road_stats = {}, {}
    for split, split_pairs_ in splits.items():
        img_dir = args.dst / f"{split}_images"
        mask_dir = args.dst / f"{split}_masks"
        for d in (img_dir, mask_dir):
            d.mkdir(parents=True, exist_ok=True)
            for stale in d.glob("*"):
                stale.unlink()

        jobs = [(sat, mask, img_dir, mask_dir, args.mode, args.tile, args.quality)
                for sat, mask in split_pairs_]
        written = road_px = total_px = 0
        errors = []
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(convert_one, job) for job in jobs]
            for done, future in enumerate(as_completed(futures), 1):
                w, r, t, err = future.result()
                written += w
                road_px += r
                total_px += t
                if err:
                    errors.append(err)
                if done % 200 == 0 or done == len(futures):
                    print(f"[prep] {split}: {done}/{len(futures)} images -> {written} tiles",
                          flush=True)
        counts[split] = written
        road_stats[split] = round(road_px / total_px, 6) if total_px else 0.0
        for err in errors[:5]:
            print(f"[prep] warning: {err}", file=sys.stderr)

    wanted["counts"] = counts
    wanted["road_pixel_fraction"] = road_stats
    manifest_path.write_text(json.dumps(wanted, indent=2))

    print(f"[prep] tiles per split: {counts}")
    print(f"[prep] road pixel fraction: {road_stats}")
    verify(args.dst)
    print(f"[prep] done -> {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
