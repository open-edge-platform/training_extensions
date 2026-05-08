# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""E2E validation: train -> test -> export FP32 -> test OV FP32 -> FP16 -> test OV FP16 -> optimize -> smoke INT8.

Mimics the application -> getitune flow:
- Config loaded via Configurator.from_recipe().
- SubsetConfig built from recipe data section (same as app's build_subset_config).
- Model instantiated via jsonargparse union type (same as app trainer).
- Engine created via create_engine() (same as app trainer).
- Training runs with recipe defaults and stops via early stopping.

Accuracy assertion compares FP32 OV vs FP16 OV through the **same**
OVEngine + torchmetrics pipeline, eliminating metric-implementation
mismatch between Ultralytics native metrics and torchmetrics.

Usage::

    python tests/integration/api/test_ultralytics_engine.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import cast

import numpy as np
import openvino
from jsonargparse import ArgumentParser, Namespace
from model_api.models import Model

from getitune.backend.lightning.models.base import DataInputParams, LightningModel
from getitune.backend.ultralytics.configurator import Configurator
from getitune.backend.ultralytics.models.base import UltralyticsModel
from getitune.config.data import SamplerConfig, SubsetConfig, TileConfig
from getitune.data.module import DataModule
from getitune.engine import create_engine
from getitune.types.export import ExportFormat
from getitune.types.precision import Precision
from getitune.types.task import TaskType

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

COCO_DATA = Path("/home/kprokofi/bench_data/detection/wgisd_merged_coco_small")
RECIPE = Path(__file__).resolve().parents[3] / "src" / "getitune" / "recipe" / "detection" / "yolo26_n.yaml"


def build_subset_config(data_config: dict, subset_name: str) -> SubsetConfig:
    """Build a SubsetConfig from the recipe's data config.

    Mirrors the application's ``build_subset_config`` helper in
    ``getitune_trainer.py`` (lines 243-250).
    """
    subset_cfg_data = deepcopy(data_config[f"{subset_name}_subset"])
    subset_cfg_data["input_size"] = data_config["input_size"]
    sampler_cfg_data = subset_cfg_data.pop("sampler", {})
    return SubsetConfig(sampler=SamplerConfig(**sampler_cfg_data), **subset_cfg_data)


def main() -> int:
    if not COCO_DATA.exists():
        logger.error(f"Dataset not found: {COCO_DATA}")
        return 1

    work_dir = Path(tempfile.mkdtemp(prefix="ultralytics_e2e_"))
    logger.info(f"Work directory: {work_dir}")

    configurator = Configurator.from_recipe(RECIPE)
    data_config = configurator.data_config
    training_config = configurator.config.get("training", {})

    train_subset = build_subset_config(data_config, "train")
    val_subset = build_subset_config(data_config, "val")
    test_subset = build_subset_config(data_config, "test")

    datamodule = DataModule(
        task=TaskType.DETECTION,
        data_root=str(COCO_DATA),
        train_subset=train_subset,
        val_subset=val_subset,
        test_subset=test_subset,
        tile_config=TileConfig(enable_tiler=False),
        input_size=tuple(data_config["input_size"]),
    )
    logger.info(
        f"DataModule: {len(datamodule.subsets)} subsets, "
        f"labels={datamodule.label_info.label_names}, "
        f"input_size={datamodule.input_size}"
    )

    model_cfg = deepcopy(configurator.config["model"])
    model_cfg["init_args"]["label_info"] = datamodule.label_info.label_names
    model_cfg["init_args"]["data_input_params"] = DataInputParams(
        input_size=cast("tuple[int, int]", datamodule.input_size),
        mean=datamodule.input_mean if datamodule.input_mean is not None else (0.0, 0.0, 0.0),
        std=datamodule.input_std if datamodule.input_std is not None else (1.0, 1.0, 1.0),
        intensity_config=datamodule.input_intensity_config,
    ).as_dict()

    model_parser = ArgumentParser()
    model_parser.add_argument("--model", type=LightningModel | UltralyticsModel)
    model = model_parser.instantiate_classes(Namespace(model=model_cfg)).get("model")
    logger.info(f"Model: {type(model).__name__}, imgsz={model.imgsz}")

    engine = create_engine(
        model=model,
        data=datamodule,
        work_dir=work_dir,
        device="auto",
        train_args=training_config,
        export_args={
            "confidence_threshold": configurator.config.get("export", {}).get("confidence_threshold", 0.001),
            "iou_threshold": configurator.config.get("export", {}).get("iou_threshold", 0.7),
        },
    )
    logger.info(f"Engine: {type(engine).__name__}")

    logger.info("TRAIN")
    train_metrics = engine.train(patience=10)
    logger.info(f"Train metrics: {train_metrics}")

    logger.info("TEST (PyTorch via Ultralytics)")
    pt_metrics = engine.test()
    pt_map50 = pt_metrics.get("val/map_50", 0.0)
    logger.info(f"PyTorch mAP50 (Ultralytics metric): {pt_map50:.4f}")
    assert pt_map50 > 0, f"Expected positive mAP50, got {pt_map50}"

    logger.info("EXPORT (OpenVINO FP32)")
    fp32_path = engine.export(export_format=ExportFormat.OPENVINO, export_precision=Precision.FP32)
    assert fp32_path.exists(), f"Exported FP32 model not found: {fp32_path}"
    logger.info(f"FP32 model: {fp32_path}")

    mapi_model = Model.create_model(str(fp32_path))
    result = mapi_model(np.zeros((640, 640, 3), dtype=np.uint8))
    assert result is not None, "FP32 ModelAPI smoke test failed"
    logger.info("FP32 ModelAPI smoke test passed")

    fp16_dir = work_dir / "fp16"
    fp16_dir.mkdir(parents=True, exist_ok=True)
    fp16_path = fp16_dir / fp32_path.name
    ov_model = openvino.Core().read_model(str(fp32_path))
    openvino.save_model(ov_model, str(fp16_path), compress_to_fp16=True)
    logger.info(f"FP16 model (from FP32 re-save): {fp16_path}")

    logger.info("TEST (OpenVINO FP32)")
    ov_fp32_engine = create_engine(model=fp32_path, data=datamodule, work_dir=str(work_dir / "ov_fp32"))
    fp32_metrics = ov_fp32_engine.test()
    logger.info(f"OV FP32 metrics: {fp32_metrics}")
    fp32_map50 = fp32_metrics.get("test/map_50", fp32_metrics.get("map_50", 0.0))

    logger.info("TEST (OpenVINO FP16)")
    ov_fp16_engine = create_engine(model=fp16_path, data=datamodule, work_dir=str(work_dir / "ov_fp16"))
    fp16_metrics = ov_fp16_engine.test()
    logger.info(f"OV FP16 metrics: {fp16_metrics}")
    fp16_map50 = fp16_metrics.get("test/map_50", fp16_metrics.get("map_50", 0.0))

    drop = fp32_map50 - fp16_map50
    logger.info(
        f"FP32 vs FP16 comparison (same torchmetrics pipeline): "
        f"FP32 mAP50={fp32_map50:.4f}, FP16 mAP50={fp16_map50:.4f}, drop={drop:.4f}"
    )
    assert abs(drop) <= 0.01, (
        f"FP32->FP16 accuracy drop {drop:.4f} exceeds 1% threshold. FP32={fp32_map50:.4f}, FP16={fp16_map50:.4f}"
    )

    logger.info("OPTIMIZE (INT8 from FP16)")
    int8_path = ov_fp16_engine.optimize()
    assert int8_path.exists(), f"Optimized model not found: {int8_path}"
    logger.info(f"INT8 model: {int8_path}")

    int8_mapi = Model.create_model(str(int8_path))
    result = int8_mapi(np.zeros((640, 640, 3), dtype=np.uint8))
    assert result is not None, "INT8 ModelAPI smoke test failed"
    logger.info("INT8 smoke test passed")

    logger.info(
        f"E2E COMPLETE: PT mAP50={pt_map50:.4f} (Ultralytics), "
        f"OV FP32 mAP50={fp32_map50:.4f}, OV FP16 mAP50={fp16_map50:.4f} (torchmetrics), "
        f"FP32->FP16 drop={drop:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
