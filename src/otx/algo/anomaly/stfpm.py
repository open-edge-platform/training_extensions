"""OTX STFPM model."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# TODO(someone): Revisit mypy errors after OTXLitModule deprecation and anomaly refactoring
# mypy: ignore-errors

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from anomalib.models.image.stfpm import Stfpm as AnomalibStfpm

from otx.core.model.anomaly import AnomalyMixin, OTXAnomaly
from otx.core.model.base import DataInputParams
from otx.core.types.label import AnomalyLabelInfo
from otx.core.types.task import OTXTaskType

if TYPE_CHECKING:
    from otx.core.types.label import LabelInfoTypes


class Stfpm(AnomalyMixin, AnomalibStfpm, OTXAnomaly):
    """OTX STFPM model.

    Args:
        layers (Sequence[str]): Feature extractor layers.
        backbone (str, optional): Feature extractor backbone. Defaults to "resnet18".
        task (OTXTaskType, optional): Task type of Anomaly Task. Defaults to OTXTaskType.ANOMALY.
        input_size (tuple[int, int], optional):
            Model input size in the order of height and width. Defaults to (256, 256)
    """

    def __init__(
        self,
        data_input_params: DataInputParams,
        label_info: LabelInfoTypes = AnomalyLabelInfo(),
        layers: Sequence[str] = ["layer1", "layer2", "layer3"],
        backbone: str = "resnet18",
        task: OTXTaskType = OTXTaskType.ANOMALY,
        **kwargs,
    ) -> None:
        self.data_input_params = data_input_params
        self.input_size = data_input_params.input_size
        self.task = OTXTaskType(task)
        super().__init__(
            backbone=backbone,
            layers=layers,
        )
