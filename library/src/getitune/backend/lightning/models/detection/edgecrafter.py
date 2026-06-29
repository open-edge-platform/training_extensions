# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EdgeCrafter model implementation for object detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from getitune.backend.lightning.exporter.base import ModelExporter
from getitune.backend.lightning.exporter.native import LightningModelExporter
from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.common.edgecrafter_mixin import EdgeCrafterMixin
from getitune.backend.lightning.models.detection.base import LightningDetectionModel
from getitune.backend.lightning.models.detection.detectors.edgecrafter import ECDETRDetector
from getitune.config.data import TileConfig
from getitune.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable
from getitune.types.export import TaskLevelExportParameters

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from getitune.backend.lightning.schedulers import LRSchedulerListCallable
    from getitune.metrics import MetricCallable
    from getitune.types.label import LabelInfoTypes


class EdgeCrafter(EdgeCrafterMixin, LightningDetectionModel):  # pyrefly: ignore[inconsistent-inheritance]
    """getitune Detection model class for EdgeCrafter.

    EdgeCrafter is a family of compact ViT-based DETR models designed for edge
    deployment via task-specialised distillation.  Four sizes are available:
    S (small), M (medium), L (large), and X (extra-large).

    Original paper / repository:
    https://github.com/Intellindust-AI-Lab/EdgeCrafter

    The model should be used with
    :class:`~getitune.backend.lightning.callbacks.aug_scheduler.AdaptiveTrainScheduling`
    and optional augmentation scheduling callbacks.

    Attributes:
        _pretrained_weights: Mapping from model-name to pretrained weight URL.
        input_size_multiplier: Patch-size aligned input divisor (16).

    Args:
        label_info: Information about the labels.
        data_input_params: Preprocessing parameters (input size, mean, std).
            When ``None`` the ``_default_preprocessing_params`` dict is used.
        model_name: One of ``"edgecrafter_{s,m,l,x}"``.
        optimizer: Optimizer callable. Defaults to :data:`DefaultOptimizerCallable`.
        scheduler: LR-scheduler callable. Defaults to :data:`DefaultSchedulerCallable`.
        metric: Metric callable. Defaults to MAP F-measure.
        multi_scale: Whether to use multi-scale training. Defaults to ``False``.
        torch_compile: Whether to use ``torch.compile``. Defaults to ``False``.
        tile_config: Tiling configuration. Defaults to disabled tiler.
    """

    _pretrained_weights: ClassVar[dict[str, str]] = {
        "edgecrafter_s": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_s.pth",
        "edgecrafter_m": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_m.pth",
        "edgecrafter_l": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_l.pth",
        "edgecrafter_x": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_x.pth",
    }

    input_size_multiplier = 16

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | dict | None = None,
        model_name: Literal[
            "edgecrafter_s",
            "edgecrafter_m",
            "edgecrafter_l",
            "edgecrafter_x",
        ] = "edgecrafter_s",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAveragePrecisionFMeasureCallable,
        multi_scale: bool = False,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        self.multi_scale = multi_scale
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

    def _create_model(self, num_classes: int | None = None) -> ECDETRDetector:
        """Create EdgeCrafter detection model.

        Args:
            num_classes: Number of classes. Defaults to ``self.num_classes``.

        Returns:
            Configured :class:`ECDETRDetector`.
        """
        num_classes = num_classes if num_classes is not None else self.num_classes
        return self._build_ec_model(num_classes, with_seg=False)

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Export parameters for EdgeCrafter detection."""
        # Use a conservative NMS IoU threshold: DETR produces one-to-one predictions
        # via Hungarian matching, so only suppress near-duplicate boxes.
        return super()._export_parameters.wrap(iou_threshold=0.8, nms_execute=True)

    @property
    def _exporter(self) -> ModelExporter:
        """Creates :class:`ModelExporter` for EdgeCrafter detection."""
        return LightningModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
            resize_mode="standard",
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration={
                "input_names": ["images"],
                "output_names": ["bboxes", "labels", "scores"],
                "dynamic_axes": {"images": {0: "batch"}},
                "dynamo": True,
                "opset_version": 18,
            },
            output_names=["bboxes", "labels", "scores"],
        )

    @property
    def _default_preprocessing_params(self) -> dict[str, DataInputParams]:
        """Default preprocessing parameters for each EdgeCrafter variant."""
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)
        return {
            "edgecrafter_s": DataInputParams(input_size=(512, 512), mean=imagenet_mean, std=imagenet_std),
            "edgecrafter_m": DataInputParams(input_size=(640, 640), mean=imagenet_mean, std=imagenet_std),
            "edgecrafter_l": DataInputParams(input_size=(640, 640), mean=imagenet_mean, std=imagenet_std),
            "edgecrafter_x": DataInputParams(input_size=(640, 640), mean=imagenet_mean, std=imagenet_std),
        }
