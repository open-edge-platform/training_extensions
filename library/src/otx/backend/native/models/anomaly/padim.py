# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX Padim model."""

# TODO(someone): Revisit mypy errors after OTXLitModule deprecation and anomaly refactoring
# mypy: ignore-errors

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from anomalib.models.image import Padim as AnomalibPadim

from otx.backend.native.models.anomaly.base import AnomalyMixin, OTXAnomaly
from otx.backend.native.models.base import DataInputParams
from otx.types.label import AnomalyLabelInfo
from otx.types.task import OTXTaskType

if TYPE_CHECKING:
    from otx.types.label import LabelInfoTypes


class Padim(AnomalyMixin, AnomalibPadim, OTXAnomaly):
    """OTX Padim model.

    Args:
        backbone (str, optional): Feature extractor backbone. Defaults to "resnet18".
        layers (list[str], optional): Feature extractor layers. Defaults to ["layer1", "layer2", "layer3"].
        pre_trained (bool, optional): Pretrained backbone. Defaults to True.
        n_features (int | None, optional): Number of features. Defaults to None.
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
        layers: list[str] = ["layer1", "layer2", "layer3"],  # noqa: B006
        pre_trained: bool = True,
        n_features: int | None = None,
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
            layers=layers,
            pre_trained=pre_trained,
            n_features=n_features,
        )
