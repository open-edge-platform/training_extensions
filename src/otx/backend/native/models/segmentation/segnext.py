# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SegNext model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from otx.backend.native.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.backend.native.models.segmentation.backbones import MSCAN
from otx.backend.native.models.segmentation.base import OTXSegmentationModel
from otx.backend.native.models.segmentation.heads import LightHamHead
from otx.backend.native.models.segmentation.losses import CrossEntropyLossWithIgnore
from otx.backend.native.models.segmentation.segmentors import BaseSegmentationModel
from otx.backend.native.models.utils.support_otx_v1 import OTXv1Helper
from otx.config.data import TileConfig
from otx.metrics.dice import SegmCallable

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch import nn

    from otx.backend.native.schedulers import LRSchedulerListCallable
    from otx.metrics import MetricCallable
    from otx.types.label import LabelInfoTypes


class SegNext(OTXSegmentationModel):
    """SegNext Model.

    Args:
        label_info (LabelInfoTypes): Information about the hierarchical labels.
        data_input_params (DataInputParams): Parameters for data input.
        model_name (Literal, optional): Name of the model. Defaults to "segnext_small".
        optimizer (OptimizerCallable, optional): Callable for the optimizer. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Callable for the learning rate scheduler.
        Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): Callable for the metric. Defaults to SegmCallable.
        torch_compile (bool, optional): Flag to indicate whether to use torch.compile. Defaults to False.
        tile_config (TileConfig, optional): Configuration for tiling. Defaults to TileConfig(enable_tiler=False).
    """

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams,
        model_name: Literal["segnext_tiny", "segnext_small", "segnext_base"] = "segnext_small",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = SegmCallable,  # type: ignore[assignment]
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ):
        super().__init__(
            label_info=label_info,
            data_input_params=data_input_params,
            model_name=model_name,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
        )

    def _create_model(self, num_classes: int | None = None) -> nn.Module:
        # initialize backbones
        num_classes = num_classes if num_classes is not None else self.num_classes

        backbone = MSCAN(model_name=self.model_name)
        decode_head = LightHamHead(model_name=self.model_name, num_classes=num_classes)
        criterion = CrossEntropyLossWithIgnore(ignore_index=self.label_info.ignore_index)  # type: ignore[attr-defined]
        return BaseSegmentationModel(
            backbone=backbone,
            decode_head=decode_head,
            criterion=criterion,
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_seg_segnext_ckpt(state_dict, add_prefix)

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for SegNext."""
        # TODO(Kirill): check PTQ removing hamburger from ignored_scope
        return {
            "ignored_scope": {
                "patterns": ["__module.model.decode_head.hamburger*"],
                "types": [
                    "Add",
                    "MVN",
                    "Divide",
                    "Multiply",
                ],
            },
        }
