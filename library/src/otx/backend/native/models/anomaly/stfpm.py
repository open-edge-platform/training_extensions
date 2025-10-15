# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX STFPM model."""

# TODO(someone): Revisit mypy errors after OTXLitModule deprecation and anomaly refactoring
# mypy: ignore-errors

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

from anomalib.models.image.stfpm import Stfpm as AnomalibStfpm

from otx.backend.native.models.anomaly.base import AnomalyMixin, OTXAnomaly
from otx.backend.native.models.base import DataInputParams
from otx.types.label import AnomalyLabelInfo
from otx.types.task import OTXTaskType

if TYPE_CHECKING:
    from otx.types.label import LabelInfoTypes


class Stfpm(AnomalyMixin, AnomalibStfpm, OTXAnomaly):
    """OTX STFPM model.

    Args:
        layers (Sequence[str]): Feature extractor layers.
        backbone (str, optional): Feature extractor backbone. Defaults to "resnet18".
        task (Literal[
                OTXTaskType.ANOMALY_CLASSIFICATION, OTXTaskType.ANOMALY_DETECTION, OTXTaskType.ANOMALY_SEGMENTATION
            ], optional): Task type of Anomaly Task. Defaults to OTXTaskType.ANOMALY_CLASSIFICATION.
        input_size (tuple[int, int], optional):
            Model input size in the order of height and width. Defaults to (256, 256)
    """

    def __init__(
        self,
        data_input_params: DataInputParams,
        label_info: LabelInfoTypes = AnomalyLabelInfo(),
        layers: Sequence[str] = ["layer1", "layer2", "layer3"],
        backbone: str = "resnet18",
        task: Literal[
            OTXTaskType.ANOMALY,
            OTXTaskType.ANOMALY_CLASSIFICATION,
            OTXTaskType.ANOMALY_DETECTION,
            OTXTaskType.ANOMALY_SEGMENTATION,
        ] = OTXTaskType.ANOMALY_CLASSIFICATION,
        **kwargs,
    ) -> None:
        self.data_input_params = data_input_params
        self.input_size = data_input_params.input_size
        self.task = OTXTaskType(task)
        super().__init__(
            backbone=backbone,
            layers=layers,
        )
