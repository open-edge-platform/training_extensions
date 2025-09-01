# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from datumaro import Dataset as DmDataset
from model_api.tilers import Tiler

from otx.backend.native.engine import OTXEngine
from otx.backend.native.models.base import OTXModel
from otx.data.module import OTXDataModule
from otx.engine import create_engine
from otx.tools.converter import TEMPLATE_ID_MAPPING
from otx.types.task import OTXTaskType
from tests.test_helpers import CommonSemanticSegmentationExporter


@pytest.mark.parametrize("recipe", pytest.RECIPE_LIST)
def test_native_ov_engine(
    recipe: str,
    tmp_path: Path,
    fxt_accelerator: str,
    fxt_target_dataset_per_task: dict,
) -> None:
    """Test the Engine.from_config functionality.

    Args:
        task (OTXTaskType): The task type.
        tmp_path (Path): The temporary path for storing training data.
        fxt_accelerator (str): The accelerator used for training.
        fxt_target_dataset_per_task (dict): A dictionary mapping tasks to target datasets.
    """
    default_models = [str(template["recipe_path"]) for template in TEMPLATE_ID_MAPPING.values() if template["default"]]
    if recipe in default_models:
        pytest.skip("Default models are checked in geti interaction tests.")
    if "mobilenet_v4" in recipe:
        pytest.skip("MobileNetV4 is not supported yet.")
    task = Path(recipe).parent.name.lower()
    tmp_path_train = tmp_path / task

    engine = OTXEngine.from_config(
        config_path=recipe,
        data_root=fxt_target_dataset_per_task[task],
        work_dir=tmp_path / task,
        device=fxt_accelerator,
    )

    # Check OTXModel & OTXDataModule
    assert isinstance(engine.model, OTXModel)
    assert isinstance(engine.datamodule, OTXDataModule)

    max_epochs = 2
    train_metric = engine.train(max_epochs=max_epochs)
    assert len(train_metric) > 0

    test_metric = engine.test()
    assert len(test_metric) > 0

    predict_result = engine.predict()
    assert len(predict_result) > 0

    # Export IR Model
    exported_model_path: Path | dict[str, Path] = engine.export()
    if isinstance(exported_model_path, Path):
        assert exported_model_path.exists()
    elif isinstance(exported_model_path, dict):
        for key, value in exported_model_path.items():
            assert value.exists(), f"{value} for {key} doesn't exist."
    else:
        AssertionError(f"Exported model path is not a Path or a dictionary of Paths: {exported_model_path}")

    # Test with IR Model and OVEngine
    ov_engine = create_engine(
        data=engine.datamodule,
        model=exported_model_path,
        work_dir=tmp_path_train / task / "export",
    )
    test_metric_from_ov_model = ov_engine.test()
    assert len(test_metric_from_ov_model) > 0

    # List of models with explain supported.
    if task not in [
        OTXTaskType.MULTI_CLASS_CLS,
        OTXTaskType.MULTI_LABEL_CLS,
        OTXTaskType.DETECTION,
        OTXTaskType.ROTATED_DETECTION,
        # TODO(Eugene): figure out why instance segmentation model fails after decoupling.
        # OTXTaskType.INSTANCE_SEGMENTATION,
    ]:
        return

    # Predict Torch model with explain
    predictions = engine.predict(explain=True)
    assert len(predictions[0].saliency_map) > 0

    # Export IR model with explain
    exported_model_with_explain = engine.export(explain=True)
    assert exported_model_with_explain.exists()

    # Infer IR Model with explain: predict
    predictions = ov_engine.predict(explain=True, checkpoint=exported_model_with_explain)
    assert len(predictions) > 0
    sal_maps_from_prediction = predictions[0].saliency_map
    assert len(sal_maps_from_prediction) > 0


@pytest.mark.parametrize("recipe", pytest.TILE_RECIPE_LIST)
def test_engine_from_tile_recipe(
    recipe: str,
    tmp_path: Path,
    fxt_accelerator: str,
    fxt_target_dataset_per_task: dict,
):
    if "detection" in recipe:
        task = OTXTaskType.DETECTION
    elif "instance_segmentation" in recipe:
        task = OTXTaskType.INSTANCE_SEGMENTATION
    elif "semantic_segmentation" in recipe:
        task = OTXTaskType.SEMANTIC_SEGMENTATION
    else:
        pytest.skip("Only Detection, Instance Segmentation, and Semantic Segmentation are supported for now.")

    data_root = fxt_target_dataset_per_task["tiling_detection"]
    if task is OTXTaskType.SEMANTIC_SEGMENTATION:
        dataset = DmDataset.import_from(path=data_root, format="coco")
        data_root = tmp_path / "tiling_detection_css"
        dataset.export(data_root, format=CommonSemanticSegmentationExporter, save_media=True)

    engine = OTXEngine.from_config(
        config_path=recipe,
        data_root=data_root,
        work_dir=tmp_path / task,
        device=fxt_accelerator,
    )
    engine.train(max_epochs=1)
    exported_model_path = engine.export()
    assert exported_model_path.exists()

    # Check OVEngine with tiling
    ov_engine = create_engine(data=engine.datamodule, model=exported_model_path)
    metric = ov_engine.test()
    assert len(metric) > 0

    # Check OVModel & OVTiler is set correctly
    ov_model = engine._auto_configurator.get_ov_model(
        model_name=exported_model_path,
    )
    assert isinstance(ov_model.model, Tiler), "Model should be an instance of Tiler"
    assert engine.datamodule.tile_config.tile_size[0] == ov_model.model.tile_size
    assert engine.datamodule.tile_config.overlap == ov_model.model.tiles_overlap


METRIC_NAME = {
    OTXTaskType.MULTI_CLASS_CLS: "val/accuracy",
}
