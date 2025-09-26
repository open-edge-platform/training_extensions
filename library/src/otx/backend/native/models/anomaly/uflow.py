# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX UFlow model."""

# mypy: ignore-errors

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from anomalib.models.image import Uflow as AnomalibUflow

from otx.backend.native.models.anomaly.base import AnomalyMixin, OTXAnomaly
from otx.backend.native.models.base import DataInputParams
from otx.types.label import AnomalyLabelInfo
from otx.types.task import OTXTaskType

if TYPE_CHECKING:
    from otx.types.label import LabelInfoTypes


class Uflow(AnomalyMixin, AnomalibUflow, OTXAnomaly):
    """OTX UFlow model.

    Args:
        label_info (LabelInfoTypes, optional): Label information. Defaults to AnomalyLabelInfo().
        backbone (str, optional): Feature extractor backbone. Defaults to "resnet18".
        flow_steps (int, optional): Number of flow steps. Defaults to 4.
        affine_clamp (float, optional): Affine clamp. Defaults to 2.0.
        affine_subnet_channels_ratio (float, optional): Affine subnet channels ratio. Defaults to 1.0.
        permute_soft (bool, optional): Whether to use soft permutation. Defaults to False.
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
        backbone: str = "resnet18",
        flow_steps: int = 4,
        affine_clamp: float = 2.0,
        affine_subnet_channels_ratio: float = 1.0,
        permute_soft: bool = False,
        task: Literal[
            OTXTaskType.ANOMALY,
            OTXTaskType.ANOMALY_CLASSIFICATION,
            OTXTaskType.ANOMALY_DETECTION,
            OTXTaskType.ANOMALY_SEGMENTATION,
        ] = OTXTaskType.ANOMALY_CLASSIFICATION,
    ) -> None:
        self.data_input_params = data_input_params
        self.input_size = data_input_params.input_size
        self.task = OTXTaskType(task)
        super().__init__(
            backbone=backbone,
            flow_steps=flow_steps,
            affine_clamp=affine_clamp,
            affine_subnet_channels_ratio=affine_subnet_channels_ratio,
            permute_soft=permute_soft,
        )
        self._setup()
