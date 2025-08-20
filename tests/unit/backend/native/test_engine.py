# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from pytest_mock import MockerFixture

from otx.backend.native.engine import OTXEngine
from otx.backend.native.models.base import DataInputParams, OTXModel
from otx.backend.native.models.classification.multiclass_models import EfficientNetMulticlassCls
from otx.types.export import OTXExportFormatType
from otx.types.precision import OTXPrecisionType


@pytest.fixture()
def fxt_engine(tmp_path) -> OTXEngine:
    return OTXEngine(
        data="tests/assets/classification_dataset",
        model="src/otx/recipe/classification/multi_class_cls/tv_mobilenet_v3_small.yaml",
        work_dir=tmp_path,
        max_epochs=9,
    )


class TestEngine:
    def test_constructor(self, tmp_path) -> None:
        # Check auto-configuration
        data_root = "tests/assets/classification_dataset"
        engine = OTXEngine(
            work_dir=tmp_path,
            data=data_root,
            model="src/otx/recipe/classification/multi_class_cls/efficientnet_b0.yaml",
        )
        assert engine.task == "MULTI_CLASS_CLS"
        assert engine.datamodule.task == "MULTI_CLASS_CLS"
        assert isinstance(engine.model, EfficientNetMulticlassCls)

        assert "default_root_dir" in engine.trainer_params
        assert engine.trainer_params["default_root_dir"] == tmp_path
        assert "accelerator" in engine.trainer_params
        assert engine.trainer_params["accelerator"] == "auto"
        assert "devices" in engine.trainer_params
        assert engine.trainer_params["devices"] == 1

    def test_model_init(self, tmp_path, mocker):
        data_root = "tests/assets/classification_dataset"
        mock_datamodule = MagicMock()
        mock_datamodule.label_info = 4321
        mock_datamodule.input_size = (1234, 1234)
        mock_datamodule.input_mean = (0.0, 0.0, 0.0)
        mock_datamodule.input_std = (1.0, 1.0, 1.0)
        mock_datamodule.task = "MULTI_CLASS_CLS"

        mocker.patch(
            "otx.tools.auto_configurator.AutoConfigurator.get_datamodule",
            return_value=mock_datamodule,
        )
        engine = OTXEngine(
            work_dir=tmp_path,
            data=data_root,
            model="src/otx/recipe/classification/multi_class_cls/efficientnet_b0.yaml",
        )

        assert engine._model.data_input_params == DataInputParams((1234, 1234), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        assert engine._model.label_info.num_classes == 4321

    def test_training_with_override_args(self, fxt_engine, mocker) -> None:
        mocker.patch("pathlib.Path.symlink_to")
        mocker.patch("otx.backend.native.engine.Trainer.fit")
        mock_seed_everything = mocker.patch("otx.backend.native.engine.seed_everything")

        assert fxt_engine._cache.args["max_epochs"] == 9

        fxt_engine.train(max_epochs=5, seed=1234)
        assert fxt_engine._cache.args["max_epochs"] == 5
        mock_seed_everything.assert_called_once_with(1234, workers=True)

    @pytest.mark.parametrize("resume", [True, False])
    def test_training_with_checkpoint(self, fxt_engine, resume: bool, mocker: MockerFixture, tmpdir) -> None:
        checkpoint = "path/to/checkpoint.ckpt"

        mock_trainer = mocker.patch("otx.backend.native.engine.Trainer")
        mock_trainer.return_value.default_root_dir = Path(tmpdir)
        mock_trainer_fit = mock_trainer.return_value.fit

        mock_chkpt_load = mocker.patch.object(fxt_engine, "_load_model_checkpoint", return_value={})
        mock_load_state_dict_incrementally = mocker.patch.object(fxt_engine.model, "load_state_dict_incrementally")

        trained_checkpoint = Path(tmpdir) / "best.ckpt"
        trained_checkpoint.touch()
        mock_trainer.return_value.checkpoint_callback.best_model_path = trained_checkpoint

        fxt_engine.train(resume=resume, checkpoint=checkpoint)

        if resume:
            assert mock_trainer_fit.call_args.kwargs.get("ckpt_path") == checkpoint
        else:
            assert "ckpt_path" not in mock_trainer_fit.call_args.kwargs

            mock_chkpt_load.assert_called_once()
            mock_load_state_dict_incrementally.assert_called_once()

    @pytest.mark.parametrize(
        "checkpoint",
        [
            "tests/assets/test_snapshots/dummy_checkpoint_cls_2.2.0.ckpt",
            "tests/assets/test_snapshots/dummy_checkpoint_hlabel_2.2.0.ckpt",
            "tests/assets/test_snapshots/dummy_checkpoint_is_2.2.0.ckpt",
            "tests/assets/test_snapshots/dummy_checkpoint_det_2.2.0.ckpt",
            "tests/assets/test_snapshots/dummy_checkpoint_anomaly_2.2.0.ckpt",
            "tests/assets/test_snapshots/dummy_checkpoint_det_2.4.5.ckpt",
        ],
    )
    def test__load_model_checkpoint(self, fxt_engine, checkpoint) -> None:
        ckpt = fxt_engine._load_model_checkpoint(checkpoint, map_location="cpu")
        assert ckpt is not None
        assert isinstance(ckpt, dict)
        assert "state_dict" in ckpt
        assert "hyper_parameters" in ckpt
        assert "label_info" in ckpt["hyper_parameters"]
        assert isinstance(ckpt["state_dict"], dict)
        assert isinstance(ckpt["optimizer_states"], dict)

        if "_2.4.5" in checkpoint:
            # simple loading with weights_only=True
            chkpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert isinstance(chkpt, dict)

    def test_test(self, fxt_engine, mocker: MockerFixture) -> None:
        checkpoint = "path/to/checkpoint.ckpt"
        mock_test = mocker.patch("otx.backend.native.engine.Trainer.test")
        _ = mocker.patch("otx.backend.native.engine.AutoConfigurator.update_ov_subset_pipeline")
        mocker.patch.object(fxt_engine, "_load_model_checkpoint", return_value={})

        mock_model = mocker.create_autospec(OTXModel)
        mocker.patch.object(fxt_engine.model, "load_state_dict", return_value=mock_model)
        # Correct label_info from the checkpoint
        mock_model.label_info = fxt_engine.datamodule.label_info
        fxt_engine.test(checkpoint=checkpoint)
        mock_test.assert_called_once()

    @pytest.mark.parametrize("explain", [True, False])
    def test_predict(self, fxt_engine, explain, mocker: MockerFixture) -> None:
        checkpoint = "path/to/checkpoint.ckpt"
        mock_predict = mocker.patch("otx.backend.native.engine.Trainer.predict")
        _ = mocker.patch("otx.backend.native.engine.AutoConfigurator.update_ov_subset_pipeline")
        mocker.patch.object(fxt_engine, "_load_model_checkpoint", return_value={})
        mock_process_saliency_maps = mocker.patch(
            "otx.backend.native.models.utils.xai_utils.process_saliency_maps_in_pred_entity",
        )

        mock_model = mocker.create_autospec(OTXModel)
        mocker.patch.object(fxt_engine.model, "load_state_dict", return_value=mock_model)

        # Correct label_info from the checkpoint
        mock_model.label_info = fxt_engine.datamodule.label_info
        fxt_engine.predict(checkpoint=checkpoint, explain=explain)
        mock_predict.assert_called_once()
        assert mock_process_saliency_maps.called == explain

    def test_exporting(self, fxt_engine, mocker) -> None:
        with pytest.raises(RuntimeError, match="To make export, checkpoint must be specified."):
            fxt_engine.export()

        mock_export = mocker.patch("otx.backend.native.engine.OTXModel.export")

        mock_load_from_checkpoint = mocker.patch.object(fxt_engine, "_load_model_checkpoint", return_value={})
        mocker.patch.object(fxt_engine.model, "load_state_dict", return_value=fxt_engine.model)

        # Fetch Checkpoint
        checkpoint = "path/to/checkpoint.ckpt"
        fxt_engine.checkpoint = checkpoint
        fxt_engine.export()
        mock_load_from_checkpoint.assert_called_once()
        mock_export.assert_called_once_with(
            output_dir=Path(fxt_engine.work_dir),
            base_name="exported_model",
            export_format=OTXExportFormatType.OPENVINO,
            precision=OTXPrecisionType.FP32,
            to_exportable_code=False,
        )

        fxt_engine.export(export_precision=OTXPrecisionType.FP16)
        mock_export.assert_called_with(
            output_dir=Path(fxt_engine.work_dir),
            base_name="exported_model",
            export_format=OTXExportFormatType.OPENVINO,
            precision=OTXPrecisionType.FP16,
            to_exportable_code=False,
        )

        fxt_engine.export(export_format=OTXExportFormatType.ONNX)
        mock_export.assert_called_with(
            output_dir=Path(fxt_engine.work_dir),
            base_name="exported_model",
            export_format=OTXExportFormatType.ONNX,
            precision=OTXPrecisionType.FP32,
            to_exportable_code=False,
        )

        fxt_engine.export(export_format=OTXExportFormatType.ONNX, export_demo_package=True)
        mock_export.assert_called_with(
            output_dir=Path(fxt_engine.work_dir),
            base_name="exported_model",
            export_format=OTXExportFormatType.ONNX,
            precision=OTXPrecisionType.FP32,
            to_exportable_code=False,
        )

    @pytest.mark.parametrize(
        "checkpoint",
        [
            "path/to/checkpoint.ckpt",
            "path/to/checkpoint.xml",
        ],
    )
    def test_explain(self, fxt_engine, checkpoint, mocker) -> None:
        mock_predict = mocker.patch("otx.backend.native.engine.Trainer.predict")
        _ = mocker.patch("otx.backend.native.engine.AutoConfigurator.update_ov_subset_pipeline")
        mocker.patch.object(fxt_engine, "_load_model_checkpoint", return_value={})
        mock_process_saliency_maps = mocker.patch(
            "otx.backend.native.models.utils.xai_utils.process_saliency_maps_in_pred_entity",
        )

        mock_model = mocker.create_autospec(OTXModel)
        mocker.patch.object(fxt_engine.model, "load_state_dict", return_value=mock_model)

        # Correct label_info from the checkpoint
        mock_model.label_info = fxt_engine.datamodule.label_info
        fxt_engine.predict(checkpoint=checkpoint, explain=True)
        mock_predict.assert_called_once()

        mock_process_saliency_maps.assert_called_once()

    def test_from_config_with_model_name(self, tmp_path) -> None:
        model_name = "efficientnet_b0"
        data_root = "tests/assets/classification_dataset"
        task_type = "MULTI_CLASS_CLS"

        overriding = {
            "data.train_subset.batch_size": 3,
            "data.test_subset.subset_name": "TESTING",
        }

        engine = OTXEngine.from_model_name(
            model_name=model_name,
            data_root=data_root,
            task=task_type,
            work_dir=tmp_path,
            **overriding,
        )

        assert engine is not None
        assert engine.datamodule.train_subset.batch_size == 3
        assert engine.datamodule.test_subset.subset_name == "TESTING"

        with pytest.raises(FileNotFoundError):
            engine = OTXEngine.from_model_name(
                model_name="wrong_model",
                task=task_type,
                data_root=data_root,
                work_dir=tmp_path,
                **overriding,
            )

    def test_from_config(self, tmp_path, mocker) -> None:
        recipe_path = "src/otx/recipe/classification/multi_class_cls/tv_mobilenet_v3_small.yaml"
        data_root = "tests/assets/classification_dataset"
        mocker.patch("pathlib.Path.symlink_to")
        mocker.patch("otx.backend.native.engine.Trainer.fit")

        overriding = {
            "data.train_subset.batch_size": 3,
            "data.test_subset.subset_name": "TESTING",
            "max_epochs": 50,
        }

        engine = OTXEngine.from_config(
            config_path=recipe_path,
            data_root=data_root,
            work_dir=tmp_path,
            **overriding,
        )

        assert engine is not None
        assert engine.datamodule.train_subset.batch_size == 3
        assert engine.datamodule.test_subset.subset_name == "TESTING"
        # test overriding train_kwargs with config
        engine.train()
        assert engine._cache.args["max_epochs"] == 50
        assert engine.trainer.max_epochs == 50
        assert not engine._cache.args["deterministic"]
        engine.train(max_epochs=100, deterministic=True)
        assert engine._cache.args["max_epochs"] == 100
        assert engine.trainer.max_epochs == 100
        assert engine._cache.args["deterministic"]

    def test_benchmark(self, fxt_engine, mocker: MockerFixture) -> None:
        checkpoint = "path/to/checkpoint.ckpt"
        mocker.patch.object(fxt_engine, "_load_model_checkpoint", return_value={})

        mock_model = mocker.create_autospec(OTXModel)
        mocker.patch.object(fxt_engine.model, "load_state_dict", return_value=mock_model)

        # Correct label_info from the checkpoint
        mock_model.label_info = fxt_engine.datamodule.label_info
        result = fxt_engine.benchmark(checkpoint=checkpoint)
        assert "latency" in result

    def test_num_devices(self, fxt_engine, tmp_path) -> None:
        assert fxt_engine.num_devices == 1
        assert fxt_engine._cache.args.get("devices") == 1

        fxt_engine.num_devices = 2
        assert fxt_engine.num_devices == 2
        assert fxt_engine._cache.args.get("devices") == 2

        data_root = "tests/assets/classification_dataset"
        engine = OTXEngine(
            work_dir=tmp_path,
            data=data_root,
            num_devices=3,
            model="src/otx/recipe/classification/multi_class_cls/efficientnet_b0.yaml",
        )
        assert engine.num_devices == 3
        assert engine._cache.args.get("devices") == 3
