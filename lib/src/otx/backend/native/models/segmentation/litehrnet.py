# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LiteHRNet model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from torch.onnx import OperatorExportTypes

from otx.backend.native.exporter.base import OTXModelExporter
from otx.backend.native.exporter.native import OTXNativeModelExporter
from otx.backend.native.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.backend.native.models.segmentation.backbones import LiteHRNetBackbone
from otx.backend.native.models.segmentation.base import OTXSegmentationModel
from otx.backend.native.models.segmentation.heads import FCNHead
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


class LiteHRNet(OTXSegmentationModel):
    """LiteHRNet Model.

    Args:
        label_info (LabelInfoTypes): Information about the hierarchical labels.
        data_input_params (DataInputParams): Parameters for data input.
        model_name (Literal, optional): Name of the model. Defaults to "lite_hrnet_18".
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
        model_name: Literal["lite_hrnet_s", "lite_hrnet_18", "lite_hrnet_x"] = "lite_hrnet_18",
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

        backbone = LiteHRNetBackbone(self.model_name)
        decode_head = FCNHead(self.model_name, num_classes=num_classes)
        criterion = CrossEntropyLossWithIgnore(ignore_index=self.label_info.ignore_index)  # type: ignore[attr-defined]
        return BaseSegmentationModel(
            backbone=backbone,
            decode_head=decode_head,
            criterion=criterion,
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_seg_lite_hrnet_ckpt(state_dict, add_prefix)

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for LiteHRNet."""
        ignored_scope = self.ignore_scope
        optim_config = {
            "advanced_parameters": {
                "activations_range_estimator_params": {
                    "min": {"statistics_type": "QUANTILE", "aggregator_type": "MIN", "quantile_outlier_prob": 1e-4},
                    "max": {"statistics_type": "QUANTILE", "aggregator_type": "MAX", "quantile_outlier_prob": 1e-4},
                },
            },
        }
        optim_config.update(ignored_scope)
        return optim_config

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration={"operator_export_type": OperatorExportTypes.ONNX_ATEN_FALLBACK},
            output_names=["preds", "feature_vector"] if self.explain_mode else None,
        )

    @property
    def ignore_scope(self) -> dict[str, Any]:
        """Get the ignored scope for LiteHRNet."""
        if self.model_name == "lite_hrnet_x":
            return {
                "ignored_scope": {
                    "patterns": ["__module.model.decode_head.aggregator/*"],
                    "names": [
                        "__module.model.backbone.stage0.0.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.0.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.0/aten::add_/Add_1",
                        "__module.model.backbone.stage0.1.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.1.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.1/aten::add_/Add_1",
                        "__module.model.backbone.stage1.0.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.0.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.0/aten::add_/Add_1",
                        "__module.model.backbone.stage1.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.0/aten::add_/Add_2",
                        "__module.model.backbone.stage1.0/aten::add_/Add_5",
                        "__module.model.backbone.stage1.1.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.1.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.1/aten::add_/Add_1",
                        "__module.model.backbone.stage1.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.1/aten::add_/Add_2",
                        "__module.model.backbone.stage1.1/aten::add_/Add_5",
                        "__module.model.backbone.stage1.2.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.2.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.2.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.2.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.2.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.2/aten::add_/Add_1",
                        "__module.model.backbone.stage1.2.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.2/aten::add_/Add_2",
                        "__module.model.backbone.stage1.2/aten::add_/Add_5",
                        "__module.model.backbone.stage1.3.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.3.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.3.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.3.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.3.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.3/aten::add_/Add_1",
                        "__module.model.backbone.stage1.3.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.3/aten::add_/Add_2",
                        "__module.model.backbone.stage1.3/aten::add_/Add_5",
                        "__module.model.backbone.stage2.0.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage2.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage2.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage2.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_3",
                        "__module.model.backbone.stage2.0.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage2.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage2.0/aten::add_/Add_1",
                        "__module.model.backbone.stage2.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage2.0/aten::add_/Add_2",
                        "__module.model.backbone.stage2.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_3",
                        "__module.model.backbone.stage2.0/aten::add_/Add_3",
                        "__module.model.backbone.stage2.0/aten::add_/Add_6",
                        "__module.model.backbone.stage2.0/aten::add_/Add_7",
                        "__module.model.backbone.stage2.0/aten::add_/Add_11",
                        "__module.model.backbone.stage2.1.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage2.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage2.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage2.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_3",
                        "__module.model.backbone.stage2.1.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage2.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage2.1/aten::add_/Add_1",
                        "__module.model.backbone.stage2.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage2.1/aten::add_/Add_2",
                        "__module.model.backbone.stage2.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_3",
                        "__module.model.backbone.stage2.1/aten::add_/Add_3",
                        "__module.model.backbone.stage2.1/aten::add_/Add_6",
                        "__module.model.backbone.stage2.1/aten::add_/Add_7",
                        "__module.model.backbone.stage2.1/aten::add_/Add_11",
                        "__module.model.decode_head.aggregator/aten::add/Add",
                        "__module.model.decode_head.aggregator/aten::add/Add_1",
                        "__module.model.decode_head.aggregator/aten::add/Add_2",
                        "__module.model.backbone.stage2.1/aten::add_/Add",
                    ],
                },
                "preset": "performance",
            }

        if self.model_name == "lite_hrnet_18":
            return {
                "ignored_scope": {
                    "patterns": ["__module.model.backbone/*"],
                    "names": [
                        "__module.model.backbone.stage0.0.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.0.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.0/aten::add_/Add_1",
                        "__module.model.backbone.stage0.1.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.1.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.1/aten::add_/Add_1",
                        "__module.model.backbone.stage1.0.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.0.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.0/aten::add_/Add_1",
                        "__module.model.backbone.stage1.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.0/aten::add_/Add_2",
                        "__module.model.backbone.stage1.0/aten::add_/Add_5",
                        "__module.model.backbone.stage1.1.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.1.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.1/aten::add_/Add_1",
                        "__module.model.backbone.stage1.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.1/aten::add_/Add_2",
                        "__module.model.backbone.stage1.1/aten::add_/Add_5",
                        "__module.model.backbone.stage1.2.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.2.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.2.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.2.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.2.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.2/aten::add_/Add_1",
                        "__module.model.backbone.stage1.2.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.2/aten::add_/Add_2",
                        "__module.model.backbone.stage1.2/aten::add_/Add_5",
                        "__module.model.backbone.stage1.3.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.3.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.3.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.3.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.3.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.3/aten::add_/Add_1",
                        "__module.model.backbone.stage1.3.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.3/aten::add_/Add_2",
                        "__module.model.backbone.stage1.3/aten::add_/Add_5",
                        "__module.model.backbone.stage2.0.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage2.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage2.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage2.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_3",
                        "__module.model.backbone.stage2.0.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage2.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage2.0/aten::add_/Add_1",
                        "__module.model.backbone.stage2.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage2.0/aten::add_/Add_2",
                        "__module.model.backbone.stage2.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_3",
                        "__module.model.backbone.stage2.0/aten::add_/Add_3",
                        "__module.model.backbone.stage2.0/aten::add_/Add_6",
                        "__module.model.backbone.stage2.0/aten::add_/Add_7",
                        "__module.model.backbone.stage2.0/aten::add_/Add_11",
                        "__module.model.backbone.stage2.1.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage2.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage2.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage2.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_3",
                        "__module.model.backbone.stage2.1.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage2.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage2.1/aten::add_/Add_1",
                        "__module.model.backbone.stage2.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage2.1/aten::add_/Add_2",
                        "__module.model.backbone.stage2.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_3",
                        "__module.model.backbone.stage2.1/aten::add_/Add_3",
                        "__module.model.backbone.stage2.1/aten::add_/Add_6",
                        "__module.model.backbone.stage2.1/aten::add_/Add_7",
                        "__module.model.backbone.stage2.1/aten::add_/Add_11",
                        "__module.model.decode_head.aggregator/aten::add/Add",
                        "__module.model.decode_head.aggregator/aten::add/Add_1",
                        "__module.model.decode_head.aggregator/aten::add/Add_2",
                        "__module.model.backbone.stage2.1/aten::add_/Add",
                    ],
                },
                "preset": "mixed",
            }

        if self.model_name == "lite_hrnet_s":
            return {
                "ignored_scope": {
                    "names": [
                        "__module.model.backbone.stage0.0.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.0.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.0/aten::add_/Add_1",
                        "__module.model.backbone.stage0.1.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.1.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.1/aten::add_/Add_1",
                        "__module.model.backbone.stage0.2.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.2.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.2.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.2.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.2/aten::add_/Add_1",
                        "__module.model.backbone.stage0.3.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.3.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.3.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage0.3.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage0.3/aten::add_/Add_1",
                        "__module.model.backbone.stage1.0.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.0.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.0.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.0/aten::add_/Add_1",
                        "__module.model.backbone.stage1.0.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.0/aten::add_/Add_2",
                        "__module.model.backbone.stage1.0/aten::add_/Add_5",
                        "__module.model.backbone.stage1.1.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.1.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.1.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.1/aten::add_/Add_1",
                        "__module.model.backbone.stage1.1.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.1/aten::add_/Add_2",
                        "__module.model.backbone.stage1.1/aten::add_/Add_5",
                        "__module.model.backbone.stage1.2.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.2.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.2.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.2.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.2.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.2/aten::add_/Add_1",
                        "__module.model.backbone.stage1.2.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.2/aten::add_/Add_2",
                        "__module.model.backbone.stage1.2/aten::add_/Add_5",
                        "__module.model.backbone.stage1.3.layers.0.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.3.layers.0.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.3.layers.0.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.3.layers.1.cross_resolution_weighting/aten::mul/Multiply",
                        "__module.model.backbone.stage1.3.layers.1.cross_resolution_weighting/aten::mul/Multiply_1",
                        "__module.model.backbone.stage1.3/aten::add_/Add_1",
                        "__module.model.backbone.stage1.3.layers.1.cross_resolution_weighting/aten::mul/Multiply_2",
                        "__module.model.backbone.stage1.3/aten::add_/Add_2",
                        "__module.model.backbone.stage1.3/aten::add_/Add_5",
                        "__module.model.decode_head.aggregator/aten::add/Add",
                        "__module.model.decode_head.aggregator/aten::add/Add_1",
                    ],
                },
                "preset": "mixed",
            }

        return {}
