# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export all getitune models with their original pre-trained weights (including head).

This script exports models to OpenVINO IR format using the ORIGINAL pre-trained weights
(e.g., COCO for detection/instance segmentation, ImageNet for classification).
The number of classes matches the pre-trained dataset:
  - Detection: 80 (COCO) or 91 (COCO with background, for RF-DETR)
  - Instance Segmentation: 80 (COCO) or 91 (RF-DETR)
  - Classification (multi-class/multi-label): 1000 (ImageNet-1K)
  - Semantic Segmentation: 2 (binary, backbone-only pre-trained)
  - Keypoint Detection: 17 (COCO keypoints)
  - Rotated Detection: 80 (COCO)

Usage:
    # Export all models (from the repository root or library/ directory):
    python export_pretrained_models.py

    # Export specific task only:
    python export_pretrained_models.py --task DETECTION

    # Export specific model only:
    python export_pretrained_models.py --task DETECTION --model yolox_s

    # Custom output directory:
    python export_pretrained_models.py --output-dir ./exported_models

    # Export to ONNX format:
    python export_pretrained_models.py --format ONNX

    # Export with FP16 precision:
    python export_pretrained_models.py --precision FP16
"""

from __future__ import annotations

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

from jsonargparse import ArgumentParser as JArgParser
from jsonargparse import Namespace

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Tasks to export (skip H_LABEL_CLS as it requires dataset-specific hierarchical label structure)
EXPORTABLE_TASKS = [
    "MULTI_CLASS_CLS",
    "MULTI_LABEL_CLS",
    "DETECTION",
    "INSTANCE_SEGMENTATION",
    "SEMANTIC_SEGMENTATION",
    "ROTATED_DETECTION",
    "KEYPOINT_DETECTION",
]


def get_recipe_paths(task: str) -> list[str]:
    """Get all recipe YAML paths for a given task, excluding OpenVINO and tile variants."""
    from getitune.backend.lightning.cli.utils import list_models
    from getitune.types.task import TaskType

    recipes = list_models(task=TaskType(task), return_recipes=True)
    # Filter out openvino_model and tile variants
    filtered = []
    for r in recipes:
        name = Path(r).stem
        if name == "openvino_model":
            continue
        if "_tile" in name:
            continue
        filtered.append(r)
    return sorted(filtered)


def load_recipe_config(recipe_path: str) -> dict:
    """Load and resolve a recipe YAML configuration."""
    from getitune.cli.utils.jsonargparse import get_configuration

    return get_configuration(recipe_path)


def get_label_info_from_config(config: dict) -> int | None:
    """Extract label_info (num_classes) from a resolved recipe config."""
    model_config = config.get("model", {})
    init_args = model_config.get("init_args", {})
    label_info = init_args.get("label_info")
    if isinstance(label_info, int):
        return label_info
    return None


def get_input_size_from_config(config: dict) -> tuple[int, int] | None:
    """Extract input_size from a resolved recipe config (checks overrides first, then data)."""
    # Check overrides first
    overrides = config.get("overrides", {})
    if isinstance(overrides, dict):
        data_overrides = overrides.get("data", {})
        if isinstance(data_overrides, dict):
            input_size = data_overrides.get("input_size")
            if input_size is not None:
                return tuple(input_size)

    # Fallback to data config
    data_config = config.get("data", {})
    if isinstance(data_config, dict):
        input_size = data_config.get("input_size")
        if input_size is not None:
            return tuple(input_size)

    return None


def create_model_from_config(config: dict, label_info: int, input_size: tuple[int, int] | None = None):
    """Create a LightningModel from config with original label_info and optional input_size.

    The model is created with the original pre-trained number of classes,
    so pre-trained weights (including the head) are loaded as-is.
    """
    from getitune.backend.lightning.models.base import LightningModel
    from getitune.utils.utils import should_pass_label_info, get_model_cls_from_config

    model_config = deepcopy(config["model"])

    # Remove non-serializable fields that get_configuration may resolve to function objects.
    # These are optional fields with defaults in the model class and are not needed for export.
    init_args = model_config.get("init_args", {})
    for key in list(init_args.keys()):
        if callable(init_args[key]) and not isinstance(init_args[key], (type, dict)):
            del init_args[key]

    # Remove pre-resolved None data_input_params so we can inject ours
    if init_args.get("data_input_params") is None:
        init_args.pop("data_input_params", None)

    # Inject data_input_params if input_size is known
    # This helps the model configure correct preprocessing for export
    if input_size is not None:
        model_config["init_args"]["data_input_params"] = {"input_size": input_size}

    model_cls = get_model_cls_from_config(Namespace(model_config))

    skip = set()
    if should_pass_label_info(model_cls):
        model_config["init_args"]["label_info"] = label_info
        skip.add("label_info")

    model_parser = JArgParser()
    model_parser.add_subclass_arguments(
        LightningModel,
        "model",
        skip=skip,
        required=False,
        fail_untyped=False,
    )
    return model_parser.instantiate_classes(Namespace(model=model_config)).get("model")


def export_model(model, output_dir: Path, export_format: str = "OPENVINO", precision: str = "FP32") -> Path:
    """Export a model with its current (pre-trained) weights."""
    from getitune.types.export import ExportFormat
    from getitune.types.precision import Precision

    fmt = ExportFormat(export_format)
    prec = Precision(precision)

    output_dir.mkdir(parents=True, exist_ok=True)
    return model.export(
        output_dir=output_dir,
        base_name="exported_model",
        export_format=fmt,
        precision=prec,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export getitune models with original pre-trained weights (including head)."
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=EXPORTABLE_TASKS,
        help="Export models for a specific task only. If not specified, all tasks are exported.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Export a specific model only (recipe stem name, e.g. 'yolox_s', 'dino_v2').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./exported_pretrained_models",
        help="Root output directory for exported models. Default: ./exported_pretrained_models",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="OPENVINO",
        choices=["OPENVINO", "ONNX"],
        help="Export format. Default: OPENVINO",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="FP32",
        choices=["FP32", "FP16"],
        help="Export precision. Default: FP32",
    )
    args = parser.parse_args()

    tasks = [args.task] if args.task else EXPORTABLE_TASKS
    output_root = Path(args.output_dir)

    logger.info("=" * 70)
    logger.info("Exporting getitune models with ORIGINAL pre-trained weights")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_root.resolve()}")
    logger.info(f"Export format: {args.format} | Precision: {args.precision}")
    logger.info(f"Tasks: {tasks}")
    logger.info("")

    results = {"success": [], "failed": [], "skipped": []}

    for task in tasks:
        logger.info(f"--- Task: {task} ---")
        recipes = get_recipe_paths(task)

        if args.model:
            recipes = [r for r in recipes if Path(r).stem == args.model]
            if not recipes:
                logger.warning(f"  Model '{args.model}' not found for task {task}")
                continue

        for recipe_path in recipes:
            model_name = Path(recipe_path).stem
            logger.info(f"  Processing: {model_name}")

            try:
                # Load and parse recipe config
                config = load_recipe_config(recipe_path)

                # Get original label_info (num_classes from pre-trained dataset)
                label_info = get_label_info_from_config(config)
                if label_info is None:
                    logger.warning(f"    SKIPPED: No label_info in recipe (requires dataset-specific labels)")
                    results["skipped"].append(f"{task}/{model_name}")
                    continue

                # Get input_size from config
                input_size = get_input_size_from_config(config)

                logger.info(f"    label_info (num_classes): {label_info}")
                if input_size:
                    logger.info(f"    input_size: {input_size}")

                # Create model with original pre-trained weights
                logger.info(f"    Creating model with original pre-trained weights...")
                model = create_model_from_config(config, label_info=label_info, input_size=input_size)

                # Export
                export_dir = output_root / task.lower() / model_name
                logger.info(f"    Exporting to: {export_dir}")
                exported_path = export_model(
                    model,
                    output_dir=export_dir,
                    export_format=args.format,
                    precision=args.precision,
                )
                logger.info(f"    SUCCESS: {exported_path}")
                results["success"].append(f"{task}/{model_name}")

            except Exception as e:
                logger.error(f"    FAILED: {e}")
                results["failed"].append(f"{task}/{model_name}: {e}")

        logger.info("")

    # Summary
    logger.info("=" * 70)
    logger.info("EXPORT SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Successful: {len(results['success'])}")
    for name in results["success"]:
        logger.info(f"    - {name}")
    if results["skipped"]:
        logger.info(f"  Skipped: {len(results['skipped'])}")
        for name in results["skipped"]:
            logger.info(f"    - {name}")
    if results["failed"]:
        logger.info(f"  Failed: {len(results['failed'])}")
        for name in results["failed"]:
            logger.info(f"    - {name}")

    if results["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
