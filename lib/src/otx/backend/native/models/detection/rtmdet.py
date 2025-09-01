# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RTMDet model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from otx.backend.native.exporter.base import OTXModelExporter
from otx.backend.native.exporter.native import OTXNativeModelExporter
from otx.backend.native.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.backend.native.models.common.losses import GIoULoss, QualityFocalLoss
from otx.backend.native.models.common.utils.assigners import DynamicSoftLabelAssigner
from otx.backend.native.models.common.utils.coders import DistancePointBBoxCoder
from otx.backend.native.models.common.utils.prior_generators import MlvlPointGenerator
from otx.backend.native.models.common.utils.samplers import PseudoSampler
from otx.backend.native.models.detection.backbones import CSPNeXt
from otx.backend.native.models.detection.base import OTXDetectionModel
from otx.backend.native.models.detection.detectors import SingleStageDetector
from otx.backend.native.models.detection.heads import RTMDetSepBNHead
from otx.backend.native.models.detection.losses import RTMDetCriterion
from otx.backend.native.models.detection.necks import CSPNeXtPAFPN
from otx.backend.native.models.utils.utils import load_checkpoint
from otx.config.data import TileConfig
from otx.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable
from otx.types.export import TaskLevelExportParameters

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.backend.native.schedulers import LRSchedulerListCallable
    from otx.metrics import MetricCallable
    from otx.types.label import LabelInfoTypes


class RTMDet(OTXDetectionModel):
    """OTX Detection model class for RTMDet.

    Attributes:
        pretrained_weights (ClassVar[dict[str, str]]): Dictionary containing URLs for pretrained weights.
        input_size_multiplier (int): Multiplier for the input size.

    Args:
        label_info (LabelInfoTypes): Information about the labels.
        data_input_params (DataInputParams): Parameters for data input.
        model_name (str, optional): Name of the model to use. Defaults to "rtmdet_tiny".
        optimizer (OptimizerCallable, optional): Callable for the optimizer. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Callable for the learning rate scheduler.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): Callable for the metric. Defaults to MeanAveragePrecisionFMeasureCallable.
        torch_compile (bool, optional): Whether to use torch compile. Defaults to False.
        tile_config (TileConfig, optional): Configuration for tiling. Defaults to TileConfig(enable_tiler=False).
    """

    pretrained_weights: ClassVar[dict[str, str]] = {
        "rtmdet_tiny": "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/object_detection/v2/rtmdet_tiny.pth",
    }

    input_size_multiplier = 32

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams,
        model_name: Literal["rtmdet_tiny"] = "rtmdet_tiny",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAveragePrecisionFMeasureCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
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

    def _create_model(self, num_classes: int | None = None) -> SingleStageDetector:
        num_classes = num_classes if num_classes is not None else self.num_classes
        train_cfg = {
            "assigner": DynamicSoftLabelAssigner(topk=13),
            "sampler": PseudoSampler(),
            "allowed_border": -1,
            "pos_weight": -1,
            "debug": False,
        }

        test_cfg = {
            "nms": {"type": "nms", "iou_threshold": 0.65},
            "score_thr": 0.001,
            "mask_thr_binary": 0.5,
            "max_per_img": 300,
            "min_bbox_size": 0,
            "nms_pre": 30000,
        }

        backbone = CSPNeXt(model_name=self.model_name)
        neck = CSPNeXtPAFPN(model_name=self.model_name)
        bbox_head = RTMDetSepBNHead(
            model_name=self.model_name,
            num_classes=num_classes,
            anchor_generator=MlvlPointGenerator(offset=0, strides=[8, 16, 32]),
            bbox_coder=DistancePointBBoxCoder(),
            train_cfg=train_cfg,  # TODO ( kirill): remove
            test_cfg=test_cfg,  # TODO ( kirill): remove
        )
        criterion = RTMDetCriterion(
            num_classes=num_classes,
            loss_cls=QualityFocalLoss(use_sigmoid=True, beta=2.0, loss_weight=1.0),
            loss_bbox=GIoULoss(loss_weight=2.0),
        )
        model = SingleStageDetector(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            criterion=criterion,
            train_cfg=train_cfg,  # TODO ( kirill): remove
            test_cfg=test_cfg,  # TODO ( kirill): remove
        )
        model.init_weights()
        load_checkpoint(model, self.pretrained_weights[self.model_name], map_location="cpu")

        return model

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
            resize_mode="fit_to_window_letterbox",
            pad_value=114,
            swap_rgb=True,
            via_onnx=True,
            onnx_export_configuration={
                "input_names": ["image"],
                "output_names": ["boxes", "labels"],
                "dynamic_axes": {
                    "image": {0: "batch"},
                    "boxes": {0: "batch", 1: "num_dets"},
                    "labels": {0: "batch", 1: "num_dets"},
                },
                "autograd_inlining": False,
            },
            output_names=["bboxes", "labels", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        return super()._export_parameters.wrap(optimization_config={"preset": "mixed"})
