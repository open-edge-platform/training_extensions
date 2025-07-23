# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import pytest
from model_api.models import Model

from otx.backend.native.models.base import OTXModel
from otx.data.module import OTXDataModule
from otx.engine import create_engine
from otx.tools.converter import GetiConfigConverter
from otx.types.export import OTXExportFormatType
from otx.types.precision import OTXPrecisionType
from otx.types.task import OTXTaskType
from tests.integration.api.geti_otx_config_utils import (
    OTXConfig,
)

if TYPE_CHECKING:
    from otx.backend.native.engine import OTXEngine

TEST_ARROW_PATH = Path(__file__).parent.parent.parent / "assets" / "geti" / "arrow_configs"
DEFAULT_GETI_CONFIG_PER_TASK = {
    # OTXTaskType.KEYPOINT_DETECTION: Not supported yet as we can't import KP dataset to Geti
    OTXTaskType.MULTI_CLASS_CLS: TEST_ARROW_PATH / "classification" / "multi_class_cls",
    OTXTaskType.MULTI_LABEL_CLS: TEST_ARROW_PATH / "classification" / "multi_label_cls",
    OTXTaskType.H_LABEL_CLS: TEST_ARROW_PATH / "classification" / "h_label_cls",
    OTXTaskType.ROTATED_DETECTION: TEST_ARROW_PATH / "detection",
    OTXTaskType.DETECTION: TEST_ARROW_PATH / "detection",
    OTXTaskType.INSTANCE_SEGMENTATION: TEST_ARROW_PATH / "detection",
    OTXTaskType.SEMANTIC_SEGMENTATION: TEST_ARROW_PATH / "semantic_segmentation",
    OTXTaskType.ANOMALY: TEST_ARROW_PATH / "anomaly",
    OTXTaskType.KEYPOINT_DETECTION: TEST_ARROW_PATH / "keypoint_detection",
}


class TestEngineAPI:
    def __init__(
        self,
        task_type: OTXTaskType,
        tmp_path: Path,
        geti_template_path: Path,
        arrow_file_path: Path,
        image_path: Path,
        tiling: bool = False,
    ):
        self.task_type = task_type
        self.tmp_path = tmp_path
        self.geti_template_path = geti_template_path
        self.arrow_file_path = arrow_file_path
        self.tiling = tiling  # In the future, we can pass the whole hyper_parameters
        self.otx_config = self._convert_config()
        self.engine, self.train_kwargs = self._instantiate_engine()
        self.image = cv2.imread(str(image_path))

    def _convert_config(
        self,
    ) -> dict:
        otx_config = OTXConfig.from_yaml_file(self.geti_template_path)

        if self.tiling:
            otx_config.hyper_parameters["dataset_preparation"]["augmentation"]["tiling"]["enable"] = True

        sub_task_type = (
            self.task_type
            if self.task_type in [OTXTaskType.MULTI_LABEL_CLS, OTXTaskType.H_LABEL_CLS]
            else OTXTaskType.MULTI_CLASS_CLS
        )
        # patch sub_task_type for Geti
        otx_config.sub_task_type = sub_task_type.value
        return otx_config.to_otx_config()

    def _instantiate_engine(self) -> tuple[OTXEngine, dict[str, Any]]:
        return GetiConfigConverter.instantiate(
            config=self.otx_config,
            work_dir=self.tmp_path,
            data_root=self.arrow_file_path,
        )

    def test_model_and_data_module(self):
        """Test the instance type of the model and the datamodule."""
        assert isinstance(self.engine.model, OTXModel)
        assert isinstance(self.engine.datamodule, OTXDataModule)

    def test_training(self):
        """Test the training process."""
        max_epochs = 2
        self.train_kwargs["max_epochs"] = max_epochs
        train_metric = self.engine.train(**self.train_kwargs)
        assert len(train_metric) > 0
        assert self.engine.checkpoint

    def test_second_round(self):
        """Test the second round of training."""
        # Load the best checkpoint from the first round
        # imitate Geti, recreate an engine
        new_config = self._convert_config()
        new_engine, train_kwargs = GetiConfigConverter.instantiate(
            config=new_config,
            work_dir=self.tmp_path / "second_round",
            data_root=self.arrow_file_path,
        )
        train_kwargs["checkpoint"] = self.engine.checkpoint

        # Check if the model is loaded correctly
        assert isinstance(new_engine.model, OTXModel)

        # sanity check for 1 epoch
        train_kwargs["max_epochs"] = 1
        train_metric = new_engine.train(**train_kwargs)
        assert len(train_metric) > 0

    def test_predictions(self):
        """Test the prediction process. This is way to check that the model is valid."""
        predictions = self.engine.predict()
        assert predictions is not None
        assert len(predictions) > 0

    def test_export_and_infer_onnx(self):
        """Test exporting the model to ONNX."""
        for precision in [OTXPrecisionType.FP16, OTXPrecisionType.FP32]:
            explain = False if self.task_type == OTXTaskType.KEYPOINT_DETECTION else precision == OTXPrecisionType.FP32
            exported_path = self.engine.export(
                export_format=OTXExportFormatType.ONNX,
                export_precision=precision,
                explain=explain,
                export_demo_package=False,
            )
            export_dir = exported_path.parent
            assert export_dir.exists()

            # Test Model API
            onnx_path = export_dir / "exported_model.onnx"
            mapi_model = Model.create_model(onnx_path)
            assert mapi_model is not None

            predictions = mapi_model(self.image)
            assert predictions is not None

            exported_path.unlink(missing_ok=True)

    def test_export_and_infer_openvino(self):
        """Test exporting the model to OpenVINO."""
        for precision in [OTXPrecisionType.FP16, OTXPrecisionType.FP32]:
            explain = False if self.task_type == OTXTaskType.KEYPOINT_DETECTION else precision == OTXPrecisionType.FP32
            exported_path = self.engine.export(
                export_format=OTXExportFormatType.OPENVINO,
                export_precision=precision,
                explain=explain,
            )
            export_dir = exported_path.parent
            assert export_dir.exists()
            assert exported_path.exists()
            assert exported_path.suffix == ".xml"

            # Test Model API
            mapi_model = Model.create_model(exported_path)
            assert mapi_model is not None

            predictions = mapi_model(self.image)
            assert predictions is not None

            exported_path.unlink(missing_ok=True)

    def test_optimize_and_infer_openvino_fp32(self):
        """Test optimizing the OpenVINO model with FP32 precision."""
        explain = self.task_type != OTXTaskType.KEYPOINT_DETECTION
        exported_path = self.engine.export(
            export_format=OTXExportFormatType.OPENVINO,
            export_precision=OTXPrecisionType.FP32,
            explain=explain,
        )
        assert exported_path.suffix == ".xml"
        # instantiate the OpenVINO engine
        ov_engine = create_engine(
            model=exported_path,
            data=self.engine.datamodule,
            work_dir=self.tmp_path,
        )
        optimized_path = ov_engine.optimize()
        assert optimized_path.exists()

        # Test Model API
        mapi_model = Model.create_model(optimized_path)
        assert mapi_model is not None

        predictions = mapi_model(self.image)
        assert predictions is not None


def test_engine_api(
    task_template: tuple[OTXTaskType, Path, bool],
    tmp_path: Path,
):
    """Test the Engine API for Geti tasks.

    Args:
        fxt_engine_api (tuple): A tuple containing the task type, template path, and tiling flag.
        tmp_path (Path): Temporary directory for testing.
    """
    # Unpack the fixture
    task, template_path, tiling = task_template

    if task not in DEFAULT_GETI_CONFIG_PER_TASK:
        pytest.skip("Only the Geti Tasks are tested to reduce unnecessary resource waste.")

    config_arrow_path = DEFAULT_GETI_CONFIG_PER_TASK[task]
    arrow_file_path = (
        config_arrow_path / "tile-datum-0-of-1.arrow" if tiling else config_arrow_path / "datum-0-of-1.arrow"
    )
    image_path = config_arrow_path / "image.jpg"

    tester = TestEngineAPI(
        task_type=task,
        tmp_path=tmp_path,
        geti_template_path=template_path,
        arrow_file_path=arrow_file_path,
        image_path=image_path,
        tiling=tiling,
    )
    tester.test_model_and_data_module()
    tester.test_training()
    tester.test_second_round()
    tester.test_predictions()
    tester.test_export_and_infer_onnx()
    tester.test_export_and_infer_openvino()
    tester.test_optimize_and_infer_openvino_fp32()
