#!/usr/bin/env python3
"""Prepare dataset, export PyTorch model to OpenVINO FP32, then optimize to INT8.

Input dataset layout (COCO format with a single 'default' split):
    <source_dataset>/
        annotations/instances_default.json
        images/default/

Output layout inside <output_dir>:
    dataset/
        annotations/
            instances_train.json   ← copy of instances_default.json
            instances_val.json     ← copy of instances_default.json
            instances_test.json    ← copy of instances_default.json
        images/
            train/                 ← copy of images/default/
            val/                   ← symlink → train/
            test/                  ← symlink → train/
    otx-workspace/
        exported_model.xml/.bin   ← FP32 OpenVINO IR
        optimized_model.xml/.bin  ← INT8 OpenVINO IR (PTQ)

Usage
-----
python export_and_optimize.py \\
    --weights      /path/to/weights.pth \\
    --source_dataset /path/to/coco_dataset \\
    --output_dir   /path/to/output \\
    [--model_name  yolox_tiny|yolox_s|yolox_l|yolox_x]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

try:
    import otx
    from otx.backend.native.engine import OTXEngine
    from otx.backend.openvino.engine import OVEngine
except ImportError:
    raise ImportError("Required OTX modules not found. Please ensure that the OTX library is installed and accessible.")

SRC_DIR = Path(otx.__file__).parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# Step 0: Dataset preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_dataset(source: Path, dest: Path) -> None:
    """Convert a single-split COCO dataset to OTX's train / val / test layout.

    Annotations are copied (the JSON file is small).
    Images are copied once to ``dest/images/train/``; ``val/`` and ``test/``
    are created as relative symlinks so disk space is not tripled.
    """
    src_ann = source / "annotations" / "instances_default.json"
    src_imgs = source / "images" / "default"

    if not src_ann.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {src_ann}\n"
            "Expected: <source_dataset>/annotations/instances_default.json"
        )
    if not src_imgs.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {src_imgs}\n"
            "Expected: <source_dataset>/images/default/"
        )

    # ── Annotations ─────────────────────────────────────────────────────────
    ann_dir = dest / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        dst = ann_dir / f"instances_{split}.json"
        if dst.exists():
            print(f"  [dataset] Annotation already exists, skipping: {dst.name}")
        else:
            shutil.copy2(src_ann, dst)
            print(f"  [dataset] Copied annotation  → {dst.relative_to(dest.parent)}")

    # ── Images ──────────────────────────────────────────────────────────────
    imgs_dir = dest / "images"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    train_dir = imgs_dir / "train"
    if train_dir.exists():
        print(f"  [dataset] images/train already exists, skipping copy.")
    else:
        img_count = sum(1 for _ in src_imgs.rglob("*") if _.is_file())
        print(f"  [dataset] Copying {img_count} images → {train_dir}  (may take a moment…)")
        shutil.copytree(src_imgs, train_dir)
        print(f"  [dataset] Images copied.")

    # val/ and test/ are symlinks to train/ to avoid duplicating data
    for split in ("val", "test"):
        link = imgs_dir / split
        if link.exists() or link.is_symlink():
            print(f"  [dataset] images/{split} already exists, skipping.")
        else:
            link.symlink_to("train")   # relative symlink: val → train
            print(f"  [dataset] Symlinked images/{split} → train/")


# ─────────────────────────────────────────────────────────────────────────────
# Steps 1 & 2: Export + Optimize
# ─────────────────────────────────────────────────────────────────────────────

def export_and_optimize(
    weights: Path,
    dataset: Path,
    work_dir: Path,
    model_name: str,
) -> Path:
    """Export checkpoint to OpenVINO FP32, then run PTQ to produce INT8 IR.

    Returns
    -------
    Path
        Path to the optimized INT8 model XML.
    """


    recipe = SRC_DIR / "otx" / "recipe" / "detection" / f"{model_name}.yaml"
    if not recipe.exists():
        available = sorted((recipe.parent).glob("*.yaml"))
        raise FileNotFoundError(
            f"Recipe not found: {recipe}\n"
            f"Available recipes in {recipe.parent}:\n"
            + "\n".join(f"  {r.stem}" for r in available)
        )

    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Checkpoint  : {weights}")
    print(f"  Dataset     : {dataset}")
    print(f"  Recipe      : {recipe}")
    print(f"  Work dir    : {work_dir}")

    # ── Step 1: Export PyTorch → OpenVINO FP32 IR ───────────────────────────
    print("\n── Step 1 / 2 : Export PyTorch → OpenVINO FP32 ─────────────────────")
    otx_engine = OTXEngine(
        data=str(dataset),
        model=str(recipe),
        checkpoint=str(weights),
        work_dir=str(work_dir),
    )
    exported_model = otx_engine.export()
    print(f"  FP32 model  : {exported_model}")

    # ── Step 2: PTQ INT8 optimization ───────────────────────────────────────
    print("\n── Step 2 / 2 : PTQ INT8 optimization ──────────────────────────────")
    ov_engine = OVEngine(
        data=str(dataset),
        model=exported_model,
        work_dir=str(work_dir),
    )
    optimized_model = ov_engine.optimize()
    print(f"  INT8 model  : {optimized_model}")

    return Path(optimized_model)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a COCO dataset, export a PyTorch checkpoint to OpenVINO FP32,\n"
            "and optimize it to INT8 via Post-Training Quantization (NNCF).\n\n"
            "Input dataset layout expected:\n"
            "  <source_dataset>/annotations/instances_default.json\n"
            "  <source_dataset>/images/default/\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to the PyTorch checkpoint (.pth or .ckpt).",
    )
    parser.add_argument(
        "--source_dataset",
        required=True,
        type=Path,
        metavar="PATH",
        help=(
            "Root of the source COCO dataset containing "
            "'annotations/instances_default.json' and 'images/default/'."
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        metavar="PATH",
        help=(
            "Directory where all outputs are written (created if absent).\n"
            "Sub-directories: dataset/, otx-workspace/."
        ),
    )
    parser.add_argument(
        "--model_name",
        default="yolox_tiny",
        choices=["yolox_tiny", "yolox_s", "yolox_l", "yolox_x"],
        help="YOLOX variant (default: yolox_tiny).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights: Path     = args.weights.resolve()
    source: Path      = args.source_dataset.resolve()
    output_dir: Path  = args.output_dir.resolve()
    model_name: str   = args.model_name

    # ── Validate inputs ──────────────────────────────────────────────────────
    errors: list[str] = []
    if not weights.exists():
        errors.append(f"Weights not found: {weights}")
    if not (source / "annotations" / "instances_default.json").exists():
        errors.append(f"Annotation not found: {source / 'annotations' / 'instances_default.json'}")
    if not (source / "images" / "default").is_dir():
        errors.append(f"Images directory not found: {source / 'images' / 'default'}")
    if errors:
        for e in errors:
            print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "dataset"
    work_dir    = output_dir / "otx-workspace"

    print("=" * 65)
    print("  OTX Export & Optimize pipeline")
    print("=" * 65)
    print(f"  Model        : {model_name}")
    print(f"  Weights      : {weights}")
    print(f"  Source data  : {source}")
    print(f"  Output dir   : {output_dir}")

    # ── Step 0: Prepare dataset ──────────────────────────────────────────────
    print("\n── Step 0 / 2 : Preparing dataset ──────────────────────────────────")
    prepare_dataset(source, dataset_dir)
    print(f"  Dataset ready: {dataset_dir}")

    # ── Steps 1 & 2: Export + Optimize ──────────────────────────────────────
    optimized_model = export_and_optimize(
        weights=weights,
        dataset=dataset_dir,
        work_dir=work_dir,
        model_name=model_name,
    )

    print("\n" + "=" * 65)
    print("  Pipeline complete.")
    print(f"  Optimized INT8 model: {optimized_model}")
    print("=" * 65)


if __name__ == "__main__":
    main()
