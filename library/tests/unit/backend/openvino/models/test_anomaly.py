# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import tempfile
from pathlib import Path

import pytest
import torch
from anomalib.metrics.min_max import MinMax
from anomalib.metrics.threshold import ManualThreshold

from otx.backend.native.models.anomaly.padim import Padim
from otx.backend.native.models.anomaly.stfpm import Stfpm
from otx.backend.native.models.anomaly.uflow import Uflow
from otx.backend.native.models.base import DataInputParams
from otx.backend.openvino.models.anomaly import OVAnomalyModel
from otx.types.export import OTXExportFormatType
from otx.types.label import AnomalyLabelInfo
from otx.types.task import OTXTaskType


class TestOVAnomalyModel:
    @pytest.fixture(
        params=[
            # "padim", Cannot export padim in this way. We will face an error.
            "stfpm",
            "uflow",
        ],
    )
    def otx_model(self, request):
        if request.param == "padim":
            model = Padim(DataInputParams((448, 448), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
        elif request.param == "stfpm":
            model = Stfpm(DataInputParams((448, 448), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
        elif request.param == "uflow":
            model = Uflow(DataInputParams((448, 448), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
            model.setup()
        else:
            raise ValueError

        # NOTE: if we do not inject those into `model`,
        # we will face errors during `model.export()` such as:
        # AttributeError: 'Stfpm' object has no attribute 'normalization_metrics'
        model.normalization_metrics = MinMax()
        model.normalization_metrics.min = torch.tensor(-1.0)
        model.normalization_metrics.max = torch.tensor(1.0)
        model.image_threshold = ManualThreshold(0.0)
        model.pixel_threshold = ManualThreshold(0.0)
        model.task = OTXTaskType.ANOMALY
        return model

    def test_label_info(self, otx_model):
        with tempfile.TemporaryDirectory() as tmpdirname:
            exported_model = otx_model.export(
                output_dir=Path(tmpdirname),
                base_name="exported_model",
                export_format=OTXExportFormatType.OPENVINO,
            )
            ov_model = OVAnomalyModel(model_path=exported_model)
            assert isinstance(ov_model.label_info, AnomalyLabelInfo)
