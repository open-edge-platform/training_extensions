# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import openvino as ov
import pytest
from pytest_mock import MockerFixture

from otx.backend.openvino.engine import OVEngine
from otx.backend.openvino.models import OVModel, OVMultilabelClassificationModel
from otx.types.label import NullLabelInfo


@pytest.fixture()
def fxt_ov_model(tmp_path, get_dummy_ov_cls_model) -> OVModel:
    ov.save_model(get_dummy_ov_cls_model, f"{tmp_path}/model.xml")
    return OVMultilabelClassificationModel(model_path=f"{tmp_path}/model.xml")


@pytest.fixture()
def fxt_engine(tmp_path, fxt_ov_model) -> OVEngine:
    data_root = "tests/assets/multilabel_classification/"

    return OVEngine(
        data=data_root,
        work_dir=tmp_path,
        model=fxt_ov_model,
    )


class TestEngine:
    def test_constructor(self, mocker, tmp_path, fxt_ov_model) -> None:
        data_root = "tests/assets/multilabel_classification/"
        engine = OVEngine(data=data_root, model=fxt_ov_model, work_dir=tmp_path)
        assert engine.datamodule.task == "MULTI_LABEL_CLS"
        assert isinstance(engine.model, OVModel)

        # init with xml path, test automatic model creation
        mocker.patch("otx.backend.openvino.engine.AutoConfigurator.get_ov_model", return_value=fxt_ov_model)
        engine = OVEngine(work_dir=tmp_path, data=data_root, model=f"{tmp_path}/model.xml")
        assert engine.model == fxt_ov_model

    def test_test(self, fxt_engine, mocker: MockerFixture) -> None:
        mocker.patch(
            "otx.backend.openvino.engine.AutoConfigurator.update_ov_subset_pipeline",
            return_value=fxt_engine.datamodule,
        )
        mock_get_ov_model = mocker.patch("otx.backend.openvino.engine.AutoConfigurator.get_ov_model")
        fxt_engine._derive_task_from_ir = MagicMock(return_value="MULTI_LABEL_CLS")
        mock_model = MagicMock()
        mock_get_ov_model.return_value = mock_model
        fxt_engine._model = fxt_engine._auto_configurator.get_ov_model("model.xml")

        # Correct label_info from the checkpoint
        mock_model.label_info = fxt_engine.datamodule.label_info
        mock_model.prepare_metric_inputs = mocker.MagicMock(return_value={"preds": [1], "target": [1]})
        fxt_engine.test(metric=MagicMock())

        mock_model.label_info = NullLabelInfo()
        # Incorrect label_info from the checkpoint
        with pytest.raises(
            ValueError,
            match="To launch a test pipeline, the label information should be same (.*)",
        ):
            fxt_engine.test()

    @pytest.mark.parametrize("explain", [True, False])
    def test_predict(self, fxt_engine, tmp_path, explain, mocker: MockerFixture) -> None:
        checkpoint = f"{tmp_path}/model.xml"
        _ = mocker.patch(
            "otx.backend.openvino.engine.AutoConfigurator.update_ov_subset_pipeline",
            return_value=fxt_engine.datamodule,
        )
        mock_process_saliency_maps = mocker.patch(
            "otx.backend.native.models.utils.xai_utils.process_saliency_maps_in_pred_entity",
        )
        fxt_engine._derive_task_from_ir = MagicMock(return_value="MULTI_LABEL_CLS")
        mocker.patch("otx.backend.openvino.engine.AutoConfigurator.get_ov_model", return_value=MagicMock())
        fxt_engine._model = fxt_engine._auto_configurator.get_ov_model("model.xml")

        # Correct label_info from the checkpoint
        fxt_engine._model.label_info = fxt_engine.datamodule.label_info
        fxt_engine.predict(explain=explain)
        assert mock_process_saliency_maps.called == explain

        fxt_engine._model.label_info = NullLabelInfo()
        # Incorrect label_info from the checkpoint
        with pytest.raises(
            ValueError,
            match="To launch a predict pipeline, the label information should be same (.*)",
        ):
            fxt_engine.predict(checkpoint=checkpoint)

    def test_optimizing_model(self, fxt_engine, mocker) -> None:
        with pytest.raises(RuntimeError, match="supports only OV IR checkpoints"):
            fxt_engine.optimize(checkpoint="path/to/model.pth")

        mocker.patch(
            "otx.backend.openvino.engine.AutoConfigurator.update_ov_subset_pipeline",
            return_value=fxt_engine.datamodule,
        )
        mock_ov_model = mocker.patch("otx.backend.openvino.engine.AutoConfigurator.get_ov_model")
        mock_model = MagicMock()
        mock_ov_model.return_value = mock_model
        fxt_engine._derive_task_from_ir = MagicMock(return_value="MULTI_LABEL_CLS")

        # Fetch Checkpoint
        fxt_engine.optimize(checkpoint="path/to/exported_model.xml")
        mock_ov_model.assert_called_once()
        mock_ov_model.return_value.optimize.assert_called_once()

        # With max_data_subset_size
        fxt_engine.optimize(max_data_subset_size=100, checkpoint="path/to/exported_model.xml")
        assert mock_ov_model.return_value.optimize.call_args[0][2]["subset_size"] == 100
