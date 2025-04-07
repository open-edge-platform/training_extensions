# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
    OTXTaskType.MULTI_CLASS_CLS: "tests/assets/geti_config_arrow/classification/multi_class_cls/datum-0-of-1.arrow",
    OTXTaskType.H_LABEL_CLS: "tests/assets/geti_config_arrow/classification/h_label_cls/datum-0-of-1.arrow",
    OTXTaskType.MULTI_LABEL_CLS: "tests/assets/geti_config_arrow/classification/multi_label_cls/datum-0-of-1.arrow",
    OTXTaskType.ROTATED_DETECTION: "tests/assets/geti_config_arrow/detection/datum-0-of-1.arrow",
    OTXTaskType.DETECTION: "tests/assets/geti_config_arrow/detection/datum-0-of-1.arrow",
    OTXTaskType.INSTANCE_SEGMENTATION: "tests/assets/geti_config_arrow/detection/datum-0-of-1.arrow",
    OTXTaskType.SEMANTIC_SEGMENTATION: "tests/assets/geti_config_arrow/semantic_segmentation/datum-0-of-1.arrow",
    OTXTaskType.ANOMALY: "tests/assets/geti_config_arrow/anomaly/datum-0-of-1.arrow",
    # OTXTaskType.KEYPOINT_DETECTION: "tests/assets/geti_config_arrow/keypoint_detection/datum-0-of-1.arrow", ->????
}


@pytest.fixture()
def fxt_trained_model(task_template, tmp_path):
    task, template_path = task_template

    arrow_path = ARROW_FILE_PATHS.get(task)
    if not arrow_path:
        pytest.skip(f"No arrow file for task: {task}")

    model_template_id, hyper_parameters = load_hyper_parameters(template_path)

    sub_task_type = (
        task if task in [OTXTaskType.MULTI_LABEL_CLS, OTXTaskType.H_LABEL_CLS] else OTXTaskType.MULTI_CLASS_CLS
    )

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

    Args:
        fxt_trained_model (tuple): Tuple containing the trained engine instance and temporary path.
    """

    engine, tmp_path = fxt_trained_model

    # OTX Test
    engine.test()

    # OTX Export
    for export_case in fxt_export_list:
        export_format = export_case["export_format"]
        precision = export_case["precision"]
        with_xai = export_case["with_xai"]
        engine.export(export_format=export_format, precision=precision, with_xai=with_xai)

        # infer exported model (only for OpenVINO)
        if export_format == ExportFormat.OPENVINO:
            engine.test(checkpoint=tmp_path / "model.bin")
            engine.explain(checkpoint=tmp_path / "model.bin")
            engine.optimize(checkpoint=tmp_path / "model.bin")
            engine.predict(checkpoint=tmp_path / "model.bin")
