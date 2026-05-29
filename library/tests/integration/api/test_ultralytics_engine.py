# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration test for the Ultralytics engine workflow.

Exercises the end-to-end getitune workflow with Ultralytics-backed models:

1. Instantiate ``UltralyticsEngine`` via Configurator + create_engine().
2. Train for 2 epochs.
3. Evaluate the model (``engine.test``).
4. Get predictions (``engine.predict``).
5. Export to OpenVINO IR and validate with ModelAPI.
6. Evaluate with OVEngine (``ov_engine.test``).
7. Get OV predictions (``ov_engine.predict``).
8. Quantize to INT8 (``ov_engine.optimize``) and validate.

Uses parquet-based datasets under ``tests/assets/``.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import NamedTuple, cast

import numpy as np
import pytest
from jsonargparse import ArgumentParser, Namespace
from model_api.models import Model

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.openvino.engine import OVEngine
from getitune.backend.ultralytics.configurator import Configurator
from getitune.backend.ultralytics.engine import UltralyticsEngine
from getitune.backend.ultralytics.models.base import UltralyticsModel
from getitune.config.data import SamplerConfig, SubsetConfig, TileConfig
from getitune.data.module import DataModule
from getitune.engine import create_engine
from getitune.types.export import ExportFormat
from getitune.types.precision import Precision
from getitune.types.task import TaskType

ASSETS_ROOT = Path(__file__).resolve().parents[2] / "assets"
RECIPE_ROOT = Path(__file__).resolve().parents[3] / "src" / "getitune" / "recipe"


class _TaskSpec(NamedTuple):
    task: TaskType
    recipe_name: str
    dataset_dir: str


_TASK_SPECS: list[_TaskSpec] = [
    _TaskSpec(
        task=TaskType.DETECTION,
        recipe_name="yolo26_n",
        dataset_dir="detection_coco",
    ),
    _TaskSpec(
        task=TaskType.INSTANCE_SEGMENTATION,
        recipe_name="yolo26_n_seg",
        dataset_dir="instance_segmentation_coco",
    ),
]


def _resolve_recipe(spec: _TaskSpec) -> Path:
    task_to_subdir = {
        TaskType.DETECTION: "detection",
        TaskType.INSTANCE_SEGMENTATION: "instance_segmentation",
    }
    subdir = task_to_subdir[spec.task]
    return RECIPE_ROOT / subdir / f"{spec.recipe_name}.yaml"


def _id_fn(spec: _TaskSpec) -> str:
    return f"{spec.task.value}-{spec.recipe_name}"


_FILTERED_TASK_SPECS: list[_TaskSpec] = [
    spec for spec in _TASK_SPECS if spec.task in getattr(pytest, "TASK_LIST", list(TaskType))
]


def _build_subset_config(data_config: dict, subset_name: str) -> SubsetConfig:
    """Build a SubsetConfig from the recipe's data config."""
    subset_cfg_data = deepcopy(data_config[f"{subset_name}_subset"])
    subset_cfg_data["input_size"] = data_config["input_size"]
    sampler_cfg_data = subset_cfg_data.pop("sampler", {})
    return SubsetConfig(sampler=SamplerConfig(**sampler_cfg_data), **subset_cfg_data)


@pytest.mark.parametrize("spec", _FILTERED_TASK_SPECS, ids=_id_fn)
def test_ultralytics_engine_workflow(
    spec: _TaskSpec,
    tmp_path: Path,
    fxt_accelerator: str,
) -> None:
    """End-to-end Ultralytics engine workflow: train -> test -> predict -> export -> OV test -> quantize."""
    data_root = ASSETS_ROOT / spec.dataset_dir
    if not data_root.exists():
        pytest.skip(f"Dataset not found at {data_root}")

    recipe = _resolve_recipe(spec)
    if not recipe.exists():
        pytest.skip(f"Recipe not found: {recipe}")

    work_dir = tmp_path / spec.task.value

    configurator = Configurator.from_recipe(recipe)
    data_config = configurator.data_config
    training_config = configurator.config.get("training", {})

    train_subset = _build_subset_config(data_config, "train")
    val_subset = _build_subset_config(data_config, "val")
    test_subset = _build_subset_config(data_config, "test")

    datamodule = DataModule(
        task=spec.task,
        data_root=str(data_root),
        train_subset=train_subset,
        val_subset=val_subset,
        test_subset=test_subset,
        tile_config=TileConfig(enable_tiler=False),
        input_size=tuple(data_config["input_size"]),
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
    model_parser.add_argument("--model", type=UltralyticsModel)
    model = model_parser.instantiate_classes(Namespace(model=model_cfg)).get("model")
    assert isinstance(model, UltralyticsModel)

    engine = create_engine(
        model=model,
        data=datamodule,
        work_dir=work_dir,
        device="auto",
        train_args=training_config,
        export_args={
            "confidence_threshold": configurator.config.get("export", {}).get("confidence_threshold", 0.25),
            "iou_threshold": configurator.config.get("export", {}).get("iou_threshold", 0.5),
        },
    )
    assert isinstance(engine, UltralyticsEngine)

    train_metrics = engine.train(max_epochs=2)
    assert len(train_metrics) > 0

    test_metrics = engine.test()
    assert len(test_metrics) > 0

    predictions = engine.predict()
    assert predictions is not None
    assert len(predictions) > 0

    ov_xml_path = engine.export(
        export_format=ExportFormat.OPENVINO,
        export_precision=Precision.FP32,
    )
    assert ov_xml_path.exists()
    assert ov_xml_path.suffix == ".xml"

    dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
    mapi_model = Model.create_model(str(ov_xml_path))
    assert mapi_model is not None
    fp32_result = mapi_model(dummy_input)
    assert fp32_result is not None

    ov_engine = create_engine(
        model=ov_xml_path,
        data=datamodule,
        work_dir=str(work_dir / "ov"),
    )
    assert isinstance(ov_engine, OVEngine)

    ov_test_metrics = ov_engine.test()
    assert len(ov_test_metrics) > 0

    ov_predictions = ov_engine.predict()
    assert ov_predictions is not None
    assert len(ov_predictions) > 0

    optimized_path = ov_engine.optimize()
    assert optimized_path.exists()

    mapi_int8_model = Model.create_model(str(optimized_path))
    assert mapi_int8_model is not None
    int8_result = mapi_int8_model(dummy_input)
    assert int8_result is not None
