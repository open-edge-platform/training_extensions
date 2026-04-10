#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Quick visual check of the full CPU + GPU augmentation pipeline.

Usage (from library/):
    python debug_aug.py [--n 20] [--out /tmp/aug_debug] [--data_root PATH]

Saves augmented images (with green bboxes drawn) to --out, one PNG per sample.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torchvision.utils as tv_utils
from torchvision import tv_tensors
from torchvision.utils import draw_bounding_boxes

from otx.config.data import SubsetConfig
from otx.data.augmentation.pipeline import CPUAugmentationPipeline, GPUAugmentationPipeline


# ── helpers ──────────────────────────────────────────────────────────────────

def _denormalize(img: torch.Tensor, mean: tuple, std: tuple) -> torch.Tensor:
    """Reverse normalization for visualization."""
    m = torch.tensor(mean, dtype=img.dtype, device=img.device).view(3, 1, 1)
    s = torch.tensor(std, dtype=img.dtype, device=img.device).view(3, 1, 1)
    return (img * s + m).clamp(0, 1)


def _save(img_chw: torch.Tensor, bboxes, out_path: Path, idx: int, prefix: str = "aug") -> None:
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
    save_file = out_path / f"{prefix}_{idx:04d}.png"
    tv_utils.save_image(img_u8.float() / 255.0, str(save_file))
    print(f"  saved {save_file}  shape={img_chw.shape}  boxes={len(bboxes) if bboxes is not None else 0}")


def _build_pipeline(data_root: Path, input_size: tuple[int, int]):
    """Build the YOLOX-S train CPU + GPU pipelines for quick inspection."""
    cpu_aug_configs = [
        {
            "class_path": "otx.data.augmentation.transforms.CachedMosaic",
            "init_args": {
                "random_pop": False,
                "max_cached_images": 20,
                "img_scale": list(input_size),
            },
        },
        {
            "class_path": "torchvision.transforms.v2.RandomResizedCrop",
            "init_args": {
                "size": list(input_size),
                "scale": [0.25, 1.0],
                "ratio": [0.75, 1.333],
                "antialias": True,
            },
        },
        {
            "class_path": "torchvision.transforms.v2.SanitizeBoundingBoxes",
            "init_args": {"min_size": 1},
        },
        # MixUp: blend with a cached sample (p=1.0 for guaranteed visibility)
        {
            "class_path": "otx.data.augmentation.transforms.CachedMixUp",
            "init_args": {"img_scale": list(input_size), "alpha": 1.5, "p": 1.0,
                          "random_pop": False, "max_cached_images": 10},
        },
    ]

    gpu_aug_configs = [
        {
            "class_path": "kornia.augmentation.ColorJiggle",
            "init_args": {"brightness": 0.125, "contrast": 0.5, "saturation": 0.5,
                          "hue": 0.05, "p": 0.5},
        },
        {
            "class_path": "kornia.augmentation.RandomHorizontalFlip",
            "init_args": {"p": 0.5},
        },
        {
            "class_path": "kornia.augmentation.RandomAffine",
            "init_args": {"degrees": 10.0, "translate": [0.1, 0.1],
                          "scale": [0.1, 2.0], "shear": [-2.0, 2.0], "p": 1.0},
        },
        {
            "class_path": "kornia.augmentation.Normalize",
            "init_args": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        },
    ]

    cpu_cfg = SubsetConfig(
        batch_size=1,
        subset_name="train",
        num_workers=0,
        input_size=input_size,
        augmentations_cpu=cpu_aug_configs,
        augmentations_gpu=[],
    )
    gpu_cfg = SubsetConfig(
        batch_size=1,
        subset_name="train",
        num_workers=0,
        input_size=input_size,
        augmentations_cpu=[],
        augmentations_gpu=gpu_aug_configs,
    )
    cpu_pipeline = CPUAugmentationPipeline.from_config(cpu_cfg)
    gpu_pipeline = GPUAugmentationPipeline.from_config(
        gpu_cfg, data_keys=["input", "bbox_xyxy"], sanitize_annotations=True,
    )
    return cpu_pipeline, gpu_pipeline


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


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise full CPU + GPU augmentations")
    parser.add_argument("--n", type=int, default=20, help="Number of samples to save")
    parser.add_argument("--out", type=Path, default=Path("/tmp/aug_debug"),
                        help="Output directory for PNGs")
    parser.add_argument("--data_root", type=Path,
                        default=Path("/home/fst/bench_data/detection/wgisd_merged_coco_small"),
                        help="Dataset root (COCO format)")
    parser.add_argument("--size", type=int, default=640, help="Input size (square)")
    args = parser.parse_args()

    input_size = (args.size, args.size)
    cpu_pipeline, gpu_pipeline = _build_pipeline(args.data_root, input_size)
    print(f"\nCPU Pipeline:\n{cpu_pipeline}\n")
    print(f"GPU Pipeline:\n{gpu_pipeline}\n")

    samples = _load_samples(args.data_root, args.n * 4)  # load more for cache warm-up

    # ── Run CPU pipeline, collect results ────────────────────────────────
    cpu_results = []
    for sample in samples:
        result = cpu_pipeline(sample)
        if result is not None:
            cpu_results.append(result)

    print(f"\nCPU pipeline produced {len(cpu_results)} results")
    print(f"Saving first {args.n} full-pipeline (CPU+GPU) images to {args.out}/\n")

    # ── Apply GPU pipeline per-sample (simulating batch=1) ──────────────
    norm_mean = gpu_pipeline.mean or (0.0, 0.0, 0.0)
    norm_std = gpu_pipeline.std or (1.0, 1.0, 1.0)

    saved = 0
    for i, result in enumerate(cpu_results):
        if saved >= args.n:
            break

        img = result.image.unsqueeze(0)  # (1, C, H, W)

        bboxes = result.bboxes.float() if hasattr(result, "bboxes") and result.bboxes is not None else torch.zeros((0, 4))
        labels = result.label if hasattr(result, "label") and result.label is not None else torch.zeros(0, dtype=torch.long)

        gpu_out = gpu_pipeline(
            images=img,
            bboxes=[bboxes],
            labels=[labels],
        )

        out_img = gpu_out["images"].squeeze(0)  # (C, H, W)
        out_bboxes = gpu_out["bboxes"][0] if gpu_out.get("bboxes") else None

        # Denormalize for visualization
        vis_img = _denormalize(out_img, norm_mean, norm_std)

        _save(vis_img, out_bboxes, args.out, i, prefix="full")
        saved += 1

    print(f"\nDone. {saved} images saved to {args.out}/")


if __name__ == "__main__":
    main()
