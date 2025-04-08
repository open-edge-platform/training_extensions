# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest
from tests.integration.geti.geti_otx_config_utils import (
    ExportFormat,
    ExportParameter,
    JobType,
    OTXConfig,
    PrecisionType,
    load_hyper_parameters,
)

from otx.core.types.task import OTXTaskType
from otx.tools.converter import ConfigConverter

ARROW_FILE_PATHS = {
    # OTXTaskType.KEYPOINT_DETECTION ---> NOT SUPPORTED
    OTXTaskType.MULTI_CLASS_CLS: "tests/assets/geti_config_arrow/classification/multi_class_cls/datum-0-of-1.arrow",
    OTXTaskType.H_LABEL_CLS: "tests/assets/geti_config_arrow/classification/h_label_cls/datum-0-of-1.arrow",
    OTXTaskType.MULTI_LABEL_CLS: "tests/assets/geti_config_arrow/classification/multi_label_cls/datum-0-of-1.arrow",
    OTXTaskType.ROTATED_DETECTION: "tests/assets/geti_config_arrow/detection/datum-0-of-1.arrow",
    OTXTaskType.DETECTION: "tests/assets/geti_config_arrow/detection/datum-0-of-1.arrow",
    OTXTaskType.INSTANCE_SEGMENTATION: "tests/assets/geti_config_arrow/detection/datum-0-of-1.arrow",
    OTXTaskType.SEMANTIC_SEGMENTATION: "tests/assets/geti_config_arrow/semantic_segmentation/datum-0-of-1.arrow",
    OTXTaskType.ANOMALY: "tests/assets/geti_config_arrow/anomaly/datum-0-of-1.arrow",
}


@pytest.fixture()
def fxt_trained_model(
    task_template: tuple[OTXTaskType, str],
    tmp_path: Path,
) -> tuple[OTXTaskType, Path]:
    """Fixture to train the model using the given task template.

    Args:
        task_template (tuple): task template defined in conftest.pytest_generate_tests.
        tmp_path (Path): Temporary path for training.

    Returns:
        tuple: Tuple containing the trained engine instance and temporary path.
    """
    task_type, template_path, tiling = task_template

    arrow_path = ARROW_FILE_PATHS.get(task_type)
    if not arrow_path:
        pytest.skip(f"Task {task_type} is not supported in the test.")

    model_template_id, hyper_parameters = load_hyper_parameters(template_path)

    if tiling:
        hyper_parameters["tiling_parameters"]["enable_tiling"]["default_value"] = True
        hyper_parameters["tiling_parameters"]["enable_tiling"]["value"] = True

    sub_task_type = (
        task_type
        if task_type in [OTXTaskType.MULTI_LABEL_CLS, OTXTaskType.H_LABEL_CLS]
        else OTXTaskType.MULTI_CLASS_CLS
    )

    # Matching geti config.json style
    otx_config = OTXConfig(
        job_type=JobType.TRAIN,
        model_template_id=model_template_id,
        hyper_parameters=hyper_parameters,
        export_parameters=[
            ExportParameter(ExportFormat.OPENVINO, PrecisionType.FP32, with_xai=True),
            ExportParameter(ExportFormat.OPENVINO, PrecisionType.FP32),
            ExportParameter(ExportFormat.OPENVINO, PrecisionType.FP16),
            ExportParameter(ExportFormat.ONNX, PrecisionType.FP32),
        ],
        optimization_type=None,
        sub_task_type=sub_task_type,
    )

    config_dict = otx_config.to_otx2_config(tmp_path)
    engine_instance, train_kwargs = ConfigConverter.instantiate(
        config=config_dict,
        data_root=arrow_path,
        work_dir=tmp_path,
    )

    train_kwargs["max_epochs"] = 2
    engine_instance.train(**train_kwargs)
    return engine_instance, tmp_path


def test_otx_e2e(fxt_trained_model, fxt_export_list):
    """Test Geti OTX E2E pipeline.

    Test the following features:
    - Training (tiling/non-tiling)
    - Testing trained PyTorch model
    - Exporting models in different formats
    - Testing exported models
    - Exporting models with OpenVINO XAI
    - Testing XAI with OpenVINO and PyTorch

    Args:
        fxt_trained_model (tuple): Tuple containing the trained engine instance and temporary path.
    """

    engine, tmp_path = fxt_trained_model

    # OTX Test
    result = engine.test(
        checkpoint=tmp_path / "best_checkpoint.ckpt",
    )

    # OTX Export
    for export_case in fxt_export_list:
        exported_model_path = engine.export(
            export_format=export_case.export_format,
            export_demo_package=export_case.export_demo_package,
        )
        assert exported_model_path.name == export_case.expected_output
        if not export_case.export_demo_package and export_case.export_format == ExportFormat.OPENVINO:
            engine.test(
                checkpoint=exported_model_path,
            )

    # OTX Test XAI
    # Supported only for classification, detection and segmentation tasks.
    if engine.task in [
        OTXTaskType.MULTI_CLASS_CLS,
        OTXTaskType.MULTI_LABEL_CLS,
        OTXTaskType.H_LABEL_CLS,
        OTXTaskType.DETECTION,
        OTXTaskType.INSTANCE_SEGMENTATION,
        OTXTaskType.SEMANTIC_SEGMENTATION,
    ]:
        # Test XAI with OpenVINO
        exported_model_path = engine.export(
            export_format=ExportFormat.OPENVINO,
            explain=True,
        )

        result = engine.explain(
            checkpoint=exported_model_path,
        )
        assert isinstance(result, list)
        assert result[0].has_xai_outputs
        assert isinstance(result[0].feature_vector, list)
        assert isinstance(result[0].feature_vector[0].shape, tuple)
        assert isinstance(result[0].saliency_map[0], dict)

        # Test XAI with PyTorch
        result = engine.explain(
            checkpoint=tmp_path / "best_checkpoint.ckpt",
        )

        assert isinstance(result, list)
        assert result[0].has_xai_outputs
        assert isinstance(result[0].feature_vector, list)
        assert isinstance(result[0].feature_vector[0].shape, tuple)
        assert isinstance(result[0].saliency_map[0], dict)
