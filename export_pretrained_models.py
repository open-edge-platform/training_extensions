# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export all getitune models with their original pre-trained weights (including head).

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
from urllib.parse import urlparse

from jsonargparse import ArgumentParser as JArgParser
from jsonargparse import Namespace

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASSIFICATION_TASKS = {"MULTI_CLASS_CLS", "MULTI_LABEL_CLS"}

# Location where the getitune backbones cache their downloaded pre-trained checkpoints.
CHECKPOINT_CACHE_DIR = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"


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
    from dataclasses import asdict

    from getitune.backend.lightning.models.base import LightningModel
    from getitune.config.data import IntensityConfig
    from getitune.utils.utils import get_model_cls_from_config, should_pass_label_info

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

    # Inject data_input_params so the export embeds correct preprocessing.
    # `intensity_config` (scale_to_unit, ÷255) is mandatory: getitune trains with
    # a two-stage pipeline.
    data_input_params: dict = {"intensity_config": asdict(IntensityConfig())}
    if input_size is not None:
        data_input_params["input_size"] = input_size
    model_config["init_args"]["data_input_params"] = data_input_params

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


def _find_head_linear(model):
    """Return the final ``nn.Linear`` of a classification model's head, or ``None``.

    Supports ``LinearClsHead`` / ``MultiLabelLinearClsHead`` (``head.fc``) and
    ``VisionTransformerClsHead`` (``head.layers.head``).
    """
    from torch import nn

    head = getattr(getattr(model, "model", None), "head", None)
    if head is None:
        return None

    fc = getattr(head, "fc", None)
    if isinstance(fc, nn.Linear):
        return fc

    layers = getattr(head, "layers", None)
    vit_head = getattr(layers, "head", None) if layers is not None else None
    if isinstance(vit_head, nn.Linear):
        return vit_head

    return None


def _last_linear(module):
    """Return the last ``nn.Linear`` submodule of ``module`` (e.g. the classifier), or ``None``."""
    from torch import nn

    last = None
    for sub in module.modules():
        if isinstance(sub, nn.Linear):
            last = sub
    return last


def _head_weights_pytorchcv_efficientnet(model_name: str):
    """Pre-trained ImageNet-1k head for pytorchcv EfficientNet backbones (``output.fc.*``)."""
    import torch
    from pytorchcv.models.common.model_store import get_model_file

    path = get_model_file(model_name=model_name, local_model_store_dir_path=str(CHECKPOINT_CACHE_DIR))
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if "output.fc.weight" not in state_dict or "output.fc.bias" not in state_dict:
        return None
    return state_dict["output.fc.weight"], state_dict["output.fc.bias"]


def _head_weights_torchvision(model_name: str):
    """Pre-trained ImageNet head for torchvision backbones (last ``nn.Linear`` of the classifier)."""
    from torchvision.models import get_model, get_model_weights

    net = get_model(name=model_name, weights=get_model_weights(model_name))
    linear = _last_linear(net)
    if linear is None:
        return None
    return linear.weight.detach().clone(), linear.bias.detach().clone()


def _head_weights_timm(model_name: str):
    """Pre-trained head for timm backbones, if the checkpoint exposes a plain ``nn.Linear`` classifier."""
    import timm
    from torch import nn

    net = timm.create_model(model_name, pretrained=True)
    classifier = net.get_classifier()
    if not isinstance(classifier, nn.Linear) or classifier.bias is None:
        return None
    return classifier.weight.detach().clone(), classifier.bias.detach().clone()


def _head_weights_vit(model_name: str):
    """Pre-trained ImageNet head for augreg ViT ``.npz`` checkpoints (``head/kernel`` / ``head/bias``).

    DINOv2 checkpoints are self-supervised and contain no classification head.
    """
    import numpy as np
    import torch
    from getitune.backend.lightning.models.classification.multiclass_models.vit import pretrained_urls

    url = pretrained_urls.get(model_name)
    if url is None or not url.endswith(".npz"):
        return None  # dinov2 (.pth) and unknown variants have no usable head

    cache_file = CHECKPOINT_CACHE_DIR / Path(urlparse(url).path).name
    if not cache_file.exists():
        return None

    weights = np.load(cache_file)
    for prefix in ("", "opt/target/", "params/"):
        kernel_key, bias_key = f"{prefix}head/kernel", f"{prefix}head/bias"
        if kernel_key in weights and bias_key in weights:
            kernel = torch.from_numpy(weights[kernel_key]).t().contiguous()  # (in, out) -> (out, in)
            bias = torch.from_numpy(weights[bias_key])
            return kernel, bias
    return None


# Maps the backbone class name to (handler, needs_model_name).
_HEAD_SOURCES = {
    "EfficientNetFeatureExtractor": _head_weights_pytorchcv_efficientnet,
    "TorchvisionBackbone": _head_weights_torchvision,
    "TimmBackbone": _head_weights_timm,
    "VisionTransformerBackbone": _head_weights_vit,
}


def recover_pretrained_head(model) -> tuple[str, str]:
    """Restore the pre-trained classification head weights onto ``model`` in-place.

    The classification backbones are feature extractors that silently drop the
    checkpoint's classifier head, after which ``_create_model()`` builds a fresh,
    randomly initialized head. This copies the original pre-trained head weights
    back onto the model's head so the exported model is accurate.

    Returns:
        A ``(status, detail)`` tuple where status is one of ``"recovered"``,
        ``"skipped"`` or ``"failed"``.
    """
    import torch

    target = _find_head_linear(model)
    if target is None:
        head = getattr(getattr(model, "model", None), "head", None)
        head_type = type(head).__name__ if head is not None else "<none>"
        return "skipped", f"unsupported head type '{head_type}' (no plain nn.Linear classifier)"

    backbone = getattr(getattr(model, "model", None), "backbone", None)
    backbone_type = type(backbone).__name__ if backbone is not None else "<none>"
    handler = _HEAD_SOURCES.get(backbone_type)
    if handler is None:
        return "skipped", f"no head-recovery handler for backbone '{backbone_type}'"

    model_name = getattr(model, "model_name", None)
    if not model_name:
        return "skipped", "model has no 'model_name'"

    try:
        source = handler(model_name)
    except Exception as exc:  # best effort: failures are surfaced in the export summary
        return "failed", f"error fetching pre-trained head: {exc}"

    if source is None:
        return "skipped", f"no compatible pre-trained head available for '{model_name}'"

    weight, bias = source
    if tuple(weight.shape) != tuple(target.weight.shape) or tuple(bias.shape) != tuple(target.bias.shape):
        return (
            "skipped",
            f"head shape mismatch (checkpoint {tuple(weight.shape)} vs model {tuple(target.weight.shape)})",
        )

    with torch.no_grad():
        target.weight.copy_(weight.to(target.weight.dtype))
        target.bias.copy_(bias.to(target.bias.dtype))
    return "recovered", f"loaded pre-trained head {tuple(weight.shape)} from '{model_name}'"


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

    results = {"success": [], "failed": [], "skipped": [], "head_not_recovered": []}

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

                # Restore the pre-trained classification head (backbones drop it on load,
                # leaving a randomly initialized head -> ~0% accuracy otherwise).
                if task in CLASSIFICATION_TASKS:
                    head_status, head_detail = recover_pretrained_head(model)
                    if head_status == "recovered":
                        logger.info(f"    Pre-trained head: {head_detail}")
                    else:  # "skipped" or "failed"
                        logger.warning(f"    Pre-trained head NOT recovered (random head): {head_detail}")
                        results["head_not_recovered"].append(f"{task}/{model_name}: {head_detail}")

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
    if results["head_not_recovered"]:
        logger.warning(
            f"  Classification models exported with a RANDOM head "
            f"(pre-trained head unavailable): {len(results['head_not_recovered'])}"
        )
        for name in results["head_not_recovered"]:
            logger.warning(f"    - {name}")

    if results["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
