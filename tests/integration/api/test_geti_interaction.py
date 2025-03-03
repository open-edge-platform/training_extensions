# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import cv2
import pytest
from model_api.models import Model

from otx.core.data.module import OTXDataModule
from otx.core.model.base import OTXModel
from otx.core.types.export import OTXExportFormatType
from otx.core.types.precision import OTXPrecisionType
from otx.core.types.task import OTXTaskType
from otx.tools.converter import ConfigConverter

if TYPE_CHECKING:
    from otx.engine.engine import Engine

TEST_PATH = Path(__file__).parent.parent.parent
DEFAULT_GETI_CONFIG_PER_TASK = {
    OTXTaskType.MULTI_CLASS_CLS: TEST_PATH / "assets" / "geti_config_arrow" / "classification" / "multi_class_cls",
    OTXTaskType.MULTI_LABEL_CLS: TEST_PATH / "assets" / "geti_config_arrow" / "classification" / "multi_label_cls",
    OTXTaskType.H_LABEL_CLS: TEST_PATH / "assets" / "geti_config_arrow" / "classification" / "h_label_cls",
}


def unzip_exportable_code(
    work_dir: Path,
    exported_path: Path,
    dst_dir: Path,
) -> Path:
    """
    Unzip exportable code.
    Copied from Geti.
    """
    with zipfile.ZipFile(exported_path, mode="r") as zfp, TemporaryDirectory(prefix=str(work_dir)) as tmpdir:
        zfp.extractall(tmpdir)
        dirpath = Path(tmpdir)

        shutil.move(dirpath / "model" / "model.xml", dst_dir / "exported_model.xml")
        shutil.move(dirpath / "model" / "model.bin", dst_dir / "exported_model.bin")

    shutil.move(exported_path, dst_dir / exported_path.name)


class TestEngineAPI:
    def __init__(
        self,
        tmp_path: Path,
        geti_config_path: Path,
        arrow_file_path: Path,
        image_path: Path,
    ):
        self.tmp_path = tmp_path
        self.geti_config_path = geti_config_path
        self.arrow_file_path = arrow_file_path
        self.otx_config = self._convert_config()
        self.engine, self.train_kwargs = self._instantiate_engine()
        self.image = cv2.imread(str(image_path))

    def _convert_config(self) -> dict:
        otx_config = ConfigConverter.convert(config_path=self.geti_config_path)
        otx_config["data"]["data_format"] = "arrow"
        otx_config["data"]["train_subset"]["subset_name"] = "TRAINING"
        otx_config["data"]["val_subset"]["subset_name"] = "VALIDATION"
        otx_config["data"]["test_subset"]["subset_name"] = "TESTING"
        return otx_config

    def _instantiate_engine(self) -> tuple[Engine, dict[str, Any]]:
        return ConfigConverter.instantiate(
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

    def test_predictions(self):
        """Test the prediction process. This is way to check that the model is valid."""
        predictions = self.engine.predict()
        assert predictions is not None
        assert len(predictions) > 0

    def test_export_and_infer_onnx(self):
        """Test exporting the model to ONNX."""
        for precision in [OTXPrecisionType.FP16, OTXPrecisionType.FP32]:
            exported_path = self.engine.export(
                export_format=OTXExportFormatType.ONNX,
                export_precision=precision,
                explain=(precision == OTXPrecisionType.FP32),
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
            exported_path = self.engine.export(
                export_format=OTXExportFormatType.OPENVINO,
                export_precision=precision,
                explain=(precision == OTXPrecisionType.FP32),
                export_demo_package=True,
            )
            export_dir = exported_path.parent
            assert export_dir.exists()

            # Test Model API
            ov_export_dir = self.tmp_path / "ov_export"
            ov_export_dir.mkdir(parents=True, exist_ok=True)
            unzip_exportable_code(
                work_dir=self.tmp_path,
                exported_path=exported_path,
                dst_dir=ov_export_dir,
            )
            xml_path = ov_export_dir / "exported_model.xml"
            mapi_model = Model.create_model(xml_path)
            assert mapi_model is not None

            predictions = mapi_model(self.image)
            assert predictions is not None

            exported_path.unlink(missing_ok=True)

    def test_optimize_and_infer_openvino_fp32(self):
        """Test optimizing the OpenVINO model with FP32 precision."""
        fp32_export_dir = self.tmp_path / "fp32_export"
        fp32_export_dir.mkdir(parents=True, exist_ok=True)
        exported_path = self.engine.export(
            export_format=OTXExportFormatType.OPENVINO,
            export_precision=OTXPrecisionType.FP32,
            explain=True,
            export_demo_package=True,
        )
        unzip_exportable_code(
            work_dir=self.tmp_path,
            exported_path=exported_path,
            dst_dir=fp32_export_dir,
        )
        optimized_path = self.engine.optimize(
            checkpoint=fp32_export_dir / "exported_model.xml",
            export_demo_package=True,
        )
        assert optimized_path.exists()

        # Test Model API
        ov_optimized_dir = self.tmp_path / "ov_optimize"
        ov_optimized_dir.mkdir(parents=True, exist_ok=True)
        unzip_exportable_code(
            work_dir=self.tmp_path,
            exported_path=optimized_path,
            dst_dir=ov_optimized_dir,
        )
        xml_path = ov_optimized_dir / "exported_model.xml"
        mapi_model = Model.create_model(xml_path)
        assert mapi_model is not None

        predictions = mapi_model(self.image)
        assert predictions is not None


@pytest.mark.parametrize("task", pytest.TASK_LIST)
def test_engine_api(task: OTXTaskType, tmp_path: Path):
    if task not in DEFAULT_GETI_CONFIG_PER_TASK:
        pytest.skip("Only the Geti Tasks are tested to reduce unnecessary resource waste.")

    config_arrow_path = DEFAULT_GETI_CONFIG_PER_TASK[task]
    geti_config_path = config_arrow_path / "config.json"
    arrow_file_path = config_arrow_path / "datum-0-of-1.arrow"
    image_path = config_arrow_path / "image.jpg"

    tester = TestEngineAPI(tmp_path, geti_config_path, arrow_file_path, image_path)
    tester.test_model_and_data_module()
    tester.test_training()
    tester.test_predictions()
    tester.test_export_and_infer_onnx()
    tester.test_export_and_infer_openvino()
    tester.test_optimize_and_infer_openvino_fp32()
