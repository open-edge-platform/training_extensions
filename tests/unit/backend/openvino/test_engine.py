# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from otx.backend.openvino.engine import OVEngine
from otx.backend.openvino.models import OVModel
from otx.core.types.label import NullLabelInfo


@pytest.fixture()
def fxt_ov_model(mocker) -> OVModel:
    mocker.patch("otx.backend.openvino.engine.OVModel._create_model", return_value=MagicMock())

    return OVModel(
        model_path="path/to/model.xml",
        model_type="Classification",
    )


@pytest.fixture()
def fxt_engine(tmp_path, fxt_ov_model) -> OVEngine:
    data_root = "tests/assets/classification_dataset"

    return OVEngine(
        datamodule=data_root,
        work_dir=tmp_path,
        model=fxt_ov_model,
    )


class TestEngine:
    def test_constructor(self, mocker, tmp_path, fxt_ov_model) -> None:
        data_root = "tests/assets/classification_dataset"
        fxt_ov_model.task = "MULTI_CLASS_CLS"
        engine = OVEngine(work_dir=tmp_path, data_root=data_root, model=fxt_ov_model)
        assert engine.datamodule.task == "MULTI_CLASS_CLS"
        assert isinstance(engine.model, OVModel)

        # init with xml path, test automatic model creation
        mock_ov_model = mocker.patch("otx.backend.openvino.engine.AutoConfigurator.get_ov_model")
        engine = OVEngine(work_dir=tmp_path, data_root=data_root, model="path/to/model.xml")
        assert mock_ov_model.called_once()
        assert engine.model == mock_ov_model

    @pytest.fixture()
    def mock_datamodule(self, mocker):
        mock_datamodule = MagicMock()
        mock_datamodule.label_info = 4321
        mock_datamodule.input_size = (1234, 1234)
        mock_datamodule.input_mean = (0.0, 0.0, 0.0)
        mock_datamodule.input_std = (1.0, 1.0, 1.0)

        return mocker.patch(
            "otx.backend.openvino.utils.auto_configurator.AutoConfigurator.get_datamodule",
            return_value=mock_datamodule,
        )

    def test_model_setter(self, fxt_engine, mocker) -> None:
        assert isinstance(fxt_engine.model, OVModel)
        mock_ov_model = mocker.patch("otx.backend.openvino.engine.AutoConfigurator.get_ov_model")
        mocker.patch_object(fxt_engine, "_derive_task_from_ir", return_value="MULTI_CLASS_CLS")
        fxt_engine.model = "path/to/model.xml"
        assert mock_ov_model.called_once()
        assert fxt_engine._auto_configurator.task == "MULTI_CLASS_CLS"

    def test_test(self, fxt_engine, mocker: MockerFixture) -> None:
        mock_get_ov_model = mocker.patch("otx.backend.openvino.engine.AutoConfigurator.get_ov_model")
        mock_model = mocker.create_autospec(OVModel)

        mock_get_ov_model.return_value = mock_model

        # Correct label_info from the checkpoint
        mock_model.label_info = fxt_engine.datamodule.label_info
        mock_model.prepare_metric_inputs = mocker.MagicMock(return_value={"preds": [1], "target": [1]})
        fxt_engine.test()

        mock_model.label_info = NullLabelInfo()
        # Incorrect label_info from the checkpoint
        with pytest.raises(
            ValueError,
            match="To launch a test pipeline, the label information should be same (.*)",
        ):
            fxt_engine.test()

    @pytest.mark.parametrize("explain", [True, False])
    def test_predict(self, fxt_engine, checkpoint, explain, mocker: MockerFixture) -> None:
        _ = mocker.patch("otx.backend.openvino.engine.AutoConfigurator.update_ov_subset_pipeline")
        mock_process_saliency_maps = mocker.patch("otx.algo.utils.xai_utils.process_saliency_maps_in_pred_entity")

        # Correct label_info from the checkpoint
        fxt_engine.model.label_info = fxt_engine.datamodule.label_info
        fxt_engine.predict(explain=explain)
        assert mock_process_saliency_maps.called == explain

        fxt_engine.model.label_info = NullLabelInfo()
        # Incorrect label_info from the checkpoint
        with pytest.raises(
            ValueError,
            match="To launch a predict pipeline, the label information should be same (.*)",
        ):
            fxt_engine.predict(checkpoint=checkpoint)

    def test_optimizing_model(self, fxt_engine, mocker) -> None:
        with pytest.raises(RuntimeError, match="supports only OV IR or ONNX checkpoints"):
            fxt_engine.optimize()

        mock_ov_model = mocker.patch("otx.backend.openvino.engine.AutoConfigurator.get_ov_model")

        # Fetch Checkpoint
        fxt_engine.optimize(checkpoint="path/to/exported_model.xml")
        mock_ov_model.assert_called_once()
        mock_ov_model.return_value.optimize.assert_called_once()

        # With max_data_subset_size
        fxt_engine.optimize(max_data_subset_size=100)
        assert mock_ov_model.return_value.optimize.call_args[0][2]["subset_size"] == 100
