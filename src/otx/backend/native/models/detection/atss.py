# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ATSS model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from otx.backend.native.exporter.base import OTXModelExporter
from otx.backend.native.exporter.native import OTXNativeModelExporter
from otx.backend.native.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.backend.native.models.common.losses import CrossEntropyLoss, CrossSigmoidFocalLoss, GIoULoss
from otx.backend.native.models.common.utils.coders import DeltaXYWHBBoxCoder
from otx.backend.native.models.common.utils.prior_generators import AnchorGenerator
from otx.backend.native.models.common.utils.samplers import PseudoSampler
from otx.backend.native.models.detection.base import OTXDetectionModel
from otx.backend.native.models.detection.detectors import SingleStageDetector
from otx.backend.native.models.detection.heads import ATSSHead
from otx.backend.native.models.detection.losses import ATSSCriterion
from otx.backend.native.models.detection.necks import FPN
from otx.backend.native.models.detection.utils.assigners import ATSSAssigner
from otx.backend.native.models.utils.support_otx_v1 import OTXv1Helper
from otx.backend.native.models.utils.utils import load_checkpoint
from otx.config.data import TileConfig
from otx.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch import nn

    from otx.backend.native.schedulers import LRSchedulerListCallable
    from otx.metrics import MetricCallable
    from otx.types.label import LabelInfoTypes


class ATSS(OTXDetectionModel):
    """OTX Detection model class for ATSS.

    Attributes:
        pretrained_weights (ClassVar[dict[str, str]]): Dictionary containing URLs for pretrained weights.

    Args:
        label_info (LabelInfoTypes): Information about the labels.
        data_input_params (DataInputParams): Parameters for data input.
        model_name (Literal, optional): Name of the model to use. Defaults to "atss_mobilenetv2".
        optimizer (OptimizerCallable, optional): Callable for the optimizer. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Callable for the learning rate scheduler.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): Callable for the metric. Defaults to MeanAveragePrecisionFMeasureCallable.
        torch_compile (bool, optional): Whether to use torch compile. Defaults to False.
        tile_config (TileConfig, optional): Configuration for tiling. Defaults to TileConfig(enable_tiler=False).
    """

    pretrained_weights: ClassVar[dict[str, str]] = {
        "atss_mobilenetv2": "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/"
        "models/object_detection/v2/mobilenet_v2-atss.pth",
        "atss_resnext101": "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/"
        "object_detection/v2/resnext101_atss_070623.pth",
    }

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams,
        model_name: Literal[
            "atss_mobilenetv2",
            "atss_resnext101",
        ] = "atss_mobilenetv2",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAveragePrecisionFMeasureCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        if model_name not in self.pretrained_weights:
            msg = f"Unsupported model: {model_name}. Supported models: {list(self.pretrained_weights.keys())}"
            raise ValueError(msg)

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
        # initialize backbones
        train_cfg = {
            "assigner": ATSSAssigner(topk=9),
            "sampler": PseudoSampler(),
            "allowed_border": -1,
            "pos_weight": -1,
            "debug": False,
        }
        test_cfg = {
            "nms": {"type": "nms", "iou_threshold": 0.6},
            "min_bbox_size": 0,
            "score_thr": 0.05,
            "max_per_img": 100,
            "nms_pre": 1000,
        }
        backbone = self._build_backbone(model_name=self.model_name)
        neck = FPN(model_name=self.model_name)
        bbox_head = ATSSHead(
            model_name=self.model_name,
            num_classes=num_classes,
            anchor_generator=AnchorGenerator(
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[8, 16, 32, 64, 128],
            ),
            bbox_coder=DeltaXYWHBBoxCoder(
                target_means=(0.0, 0.0, 0.0, 0.0),
                target_stds=(0.1, 0.1, 0.2, 0.2),
            ),
            train_cfg=train_cfg,  # TODO (Kirill): remove
            test_cfg=test_cfg,  # TODO (Kirill): remove
        )
        criterion = ATSSCriterion(
            num_classes=num_classes,
            bbox_coder=DeltaXYWHBBoxCoder(
                target_means=(0.0, 0.0, 0.0, 0.0),
                target_stds=(0.1, 0.1, 0.2, 0.2),
            ),
            loss_cls=CrossSigmoidFocalLoss(
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
            ),
            loss_bbox=GIoULoss(loss_weight=2.0),
            loss_centerness=CrossEntropyLoss(use_sigmoid=True, loss_weight=1.0),
        )
        model = SingleStageDetector(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            criterion=criterion,
            train_cfg=train_cfg,  # TODO (Kirill): remove
            test_cfg=test_cfg,  # TODO (Kirill): remove
        )
        model.init_weights()
        load_checkpoint(model, self.pretrained_weights[self.model_name], map_location="cpu")

        return model

    def _build_backbone(self, model_name: str) -> nn.Module:
        if "mobilenetv2" in model_name:
            from otx.backend.native.models.common.backbones import build_model_including_pytorchcv

            return build_model_including_pytorchcv(
                cfg={
                    "type": "mobilenetv2_w1",
                    "out_indices": [2, 3, 4, 5],
                    "frozen_stages": -1,
                    "norm_eval": False,
                    "pretrained": True,
                },
            )

        if "resnext101" in model_name:
            from otx.backend.native.models.common.backbones import ResNeXt

            return ResNeXt(
                depth=101,
                groups=64,
                frozen_stages=1,
                init_cfg={"type": "Pretrained", "checkpoint": "open-mmlab://resnext101_64x4d"},
            )

        msg = f"Unknown backbone name: {model_name}"
        raise ValueError(msg)

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
            via_onnx=True,  # Currently ATSS should be exported through ONNX
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

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_det_ckpt(state_dict, add_prefix)
