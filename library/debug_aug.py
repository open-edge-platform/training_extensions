#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Quick visual check of the CPU augmentation pipeline.

Usage (from library/):
    python debug_aug.py [--n 10] [--out /tmp/aug_debug] [--data_root PATH]

Saves augmented images (with green bboxes drawn) to --out, one PNG per sample.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torchvision.utils as tv_utils
from torchvision.utils import draw_bounding_boxes

from otx.config.data import SubsetConfig
from otx.data.augmentation.pipeline import CPUAugmentationPipeline


# ── helpers ──────────────────────────────────────────────────────────────────

def _save(img_chw: torch.Tensor, bboxes, out_path: Path, idx: int) -> None:
    """Save one CHW float [0,1] tensor as a PNG with optional bbox overlay."""
    img_u8 = (img_chw.clamp(0, 1) * 255).to(torch.uint8).cpu()
    if bboxes is not None and len(bboxes) > 0:
        h, w = img_u8.shape[-2], img_u8.shape[-1]
        boxes = bboxes.float().cpu().clone()
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w - 1)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h - 1)
        # keep only valid boxes
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        if valid.any():
            img_u8 = draw_bounding_boxes(img_u8, boxes[valid], colors="green", width=2)
    out_path.mkdir(parents=True, exist_ok=True)
    save_file = out_path / f"aug_{idx:04d}.png"
    tv_utils.save_image(img_u8.float() / 255.0, str(save_file))
    print(f"  saved {save_file}  shape={img_chw.shape}  boxes={len(bboxes) if bboxes is not None else 0}")


def _build_pipeline(data_root: Path, input_size: tuple[int, int]) -> CPUAugmentationPipeline:
    """Build the YOLOX-S train CPU pipeline for quick inspection."""
    aug_configs = [
        {
            "class_path": "otx.data.augmentation.transforms.Resize",
            "init_args": {"size": list(input_size), "keep_aspect_ratio": True},
        },
        {
            "class_path": "otx.data.augmentation.transforms.CachedMosaic",
            "init_args": {
                "random_pop": False,
                "max_cached_images": 20,
                "img_scale": list(input_size),
                "scaling_ratio_range": [0.1, 2.0],
            },
        },
        # MixUp: blend with a cached sample (p=1.0 for guaranteed visibility)
        {
            "class_path": "otx.data.augmentation.transforms.CachedMixUp",
            "init_args": {"img_scale": list(input_size), "mix_ratio": 0.5, "p": 1.0,
                          "random_pop": False, "max_cached_images": 10},
        },
        {
            "class_path": "torchvision.transforms.v2.SanitizeBoundingBoxes",
            "init_args": {"min_size": 1},
        },
    ]
    cfg = SubsetConfig(
        batch_size=1,
        subset_name="train",
        num_workers=0,
        input_size=input_size,
        augmentations_cpu=aug_configs,
        augmentations_gpu=[],
    )
    return CPUAugmentationPipeline.from_config(cfg)


def _load_samples(data_root: Path, n: int):
    """Load up to n raw OTXSamples from a detection dataset via OTX internals."""
    from datumaro.experimental.export_import import import_dataset  # noqa: PLC0415
    from datumaro.experimental.fields import Subset  # noqa: PLC0415

    from otx.config.data import SubsetConfig  # noqa: PLC0415
    from otx.data.factory import OTXDatasetFactory  # noqa: PLC0415
    from otx.types.task import OTXTaskType  # noqa: PLC0415

    print(f"Loading dataset from {data_root}")
    dm_dataset = import_dataset(str(data_root))

    dm_subset = dm_dataset.filter_by_subset(Subset.TRAINING)
    if len(dm_subset) == 0:
        dm_subset = dm_dataset.filter_by_subset(Subset.VALIDATION)
    print(f"  datumaro subset size: {len(dm_subset)}")

    cfg = SubsetConfig(
        batch_size=1,
        subset_name="train",
        num_workers=0,
        augmentations_cpu=[],  # no transforms — we apply the pipeline manually
        augmentations_gpu=[],
    )
    otx_ds = OTXDatasetFactory.create(
        task=OTXTaskType.DETECTION,
        dm_subset=dm_subset,
        cfg_subset=cfg,
    )
    samples = []
    for i in range(min(n, len(otx_ds))):
        s = otx_ds[i]
        if s is not None:
            samples.append(s)
    print(f"  loaded {len(samples)} samples")
    return samples

    cfg = SubsetConfig(
        batch_size=1,
        subset_name="train",
        num_workers=0,
        augmentations_cpu=[],  # no transforms — we apply the pipeline manually
        augmentations_gpu=[],
    )
    otx_ds = OTXDatasetFactory.create(
        task=OTXTaskType.DETECTION,
        dm_subset=dm_subset,
        cfg_subset=cfg,
    )
    samples = []
    for i in range(min(n, len(otx_ds))):
        s = otx_ds[i]
        if s is not None:
            samples.append(s)
    print(f"  loaded {len(samples)} samples")
    return samples


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise CPU augmentations")
    parser.add_argument("--n", type=int, default=10, help="Number of samples to save")
    parser.add_argument("--out", type=Path, default=Path("/tmp/aug_debug"),
                        help="Output directory for PNGs")
    parser.add_argument("--data_root", type=Path,
                        default=Path("/home/fst/bench_data/detection/wgisd_merged_coco_small"),
                        help="Dataset root (COCO format)")
    parser.add_argument("--size", type=int, default=640, help="Input size (square)")
    args = parser.parse_args()

    input_size = (args.size, args.size)
    pipeline = _build_pipeline(args.data_root, input_size)
    print(f"\nPipeline:\n{pipeline}\n")

    samples = _load_samples(args.data_root, args.n * 4)  # load more for cache warm-up

    print(f"\nRunning pipeline on {len(samples)} samples, saving first {args.n} results to {args.out}/\n")
    saved = 0
    for i, sample in enumerate(samples):
        result = pipeline(sample)
        if result is None:
            continue
        _save(result.image, getattr(result, "bboxes", None), args.out, i)
        saved += 1
        if saved >= args.n:
            break

    print(f"\nDone. {saved} images saved to {args.out}/")


if __name__ == "__main__":
    main()
