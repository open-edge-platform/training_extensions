# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RF-DETR model implementations for getitune.

RF-DETR is a state-of-the-art real-time object detector from Roboflow based on
DINOv2 backbone with a lightweight DETR decoder.
Original implementation: https://github.com/roboflow/rf-detr
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from torch.export import Dim

<<<<<<<< HEAD:library/src/getitune/backend/lightning/models/detection/rfdetr.py
from getitune.backend.lightning.exporter.base import ModelExporter
from getitune.backend.lightning.exporter.native import LightningModelExporter
from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.common.rfdetr_mixin import RFDETRMixin
from getitune.backend.lightning.models.detection.base import LightningDetectionModel
from getitune.backend.lightning.models.detection.detectors.rfdetr import RFDETRDetector
========
from getitune.backend.lightning.exporter.base import ModelExporter
from getitune.backend.lightning.exporter.native import LightningModelExporter
from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.common.rfdetr_mixin import RFDETRMixin
from getitune.backend.lightning.models.detection.base import LightningDetectionModel
from getitune.backend.lightning.models.detection.detectors.rfdetr import RFDETRDetector
>>>>>>>> develop:library/src/getitune/backend/native/models/detection/rfdetr.py
from getitune.config.data import TileConfig
from getitune.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

<<<<<<<< HEAD:library/src/getitune/backend/lightning/models/detection/rfdetr.py
    from getitune.backend.lightning.schedulers import LRSchedulerListCallable
========
    from getitune.backend.lightning.schedulers import LRSchedulerListCallable
>>>>>>>> develop:library/src/getitune/backend/native/models/detection/rfdetr.py
    from getitune.metrics import MetricCallable
    from getitune.types.label import LabelInfoTypes


class RFDETR(RFDETRMixin, LightningDetectionModel):  # pyrefly: ignore[inconsistent-inheritance]
    """getitune Detection model class for RF-DETR.

    RF-DETR (Real-time Fast DETR) is a state-of-the-art object detector from Roboflow
    that combines a DINOv2 backbone with a lightweight DETR decoder for real-time
    object detection.

    This implementation uses the rfdetr Python package for the core model components.

    Args:
        label_info: Information about the labels.
        data_input_params: Parameters for image data preprocessing.
        model_name: Name of the model variant to use.
        optimizer: Callable for the optimizer.
        scheduler: Callable for the learning rate scheduler.
        metric: Callable for the metric.
        multi_scale: Whether to use multi-scale training.
        torch_compile: Whether to use torch compile.
        tile_config: Configuration for tiling.
        max_total_objects_per_batch: Maximum total number of ground-truth objects
            allowed across all images in a training batch. When set, dense
            batches are capped to this budget to prevent OOM errors caused by
            RF-DETR's group-DETR decoder architecture. Objects are kept by
            area (largest first). Set to ``None`` to disable. Only active
            during training.
        gradient_checkpointing: Whether to enable gradient checkpointing to
            reduce GPU memory usage at the cost of slower training.

    Note:
        RF-DETR uses different patch sizes for different model variants:
        - nano, small, medium: patch_size=16
        - base, large: patch_size=14
        Input sizes must be compatible with patch_size * num_windows.
    """

    _pretrained_weights: ClassVar[dict[str, str]] = {
        "rfdetr_base": "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
        "rfdetr_large": "https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth",
        "rfdetr_nano": "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
        "rfdetr_small": "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
        "rfdetr_medium": "https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth",
    }

    _model_class_mapping: ClassVar[dict[str, type]] = {
        "rfdetr_base": RFDETRBase,
        "rfdetr_large": RFDETRLarge,
        "rfdetr_medium": RFDETRMedium,
        "rfdetr_nano": RFDETRNano,
        "rfdetr_small": RFDETRSmall,
    }

    input_size_multiplier = 8

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | dict | None = None,
        model_name: Literal[
            "rfdetr_nano",
            "rfdetr_small",
            "rfdetr_base",
            "rfdetr_medium",
            "rfdetr_large",
        ] = "rfdetr_base",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAveragePrecisionFMeasureCallable,
        multi_scale: bool = False,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
        max_total_objects_per_batch: int | None = None,
        gradient_checkpointing: bool = False,
    ) -> None:
        self.multi_scale = multi_scale
        self.max_total_objects_per_batch = max_total_objects_per_batch
        self.gradient_checkpointing = gradient_checkpointing
        super().__init__(
            label_info=label_info,
            data_input_params=data_input_params,
            model_name=model_name,  # type: ignore[arg-type]
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
        )

    def _create_model(self, num_classes: int | None = None) -> RFDETRDetector:  # pyrefly: ignore[bad-override]
        """Create RF-DETR model using rfdetr package.

        Args:
            num_classes: Number of classes for detection.

        Returns:
            RFDETRDetector model instance.
        """
        num_classes = num_classes if num_classes is not None else self.num_classes
        return self._build_rfdetr_model(
            num_classes,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    @property
    def _exporter(self) -> ModelExporter:
        """Creates ModelExporter object for model export."""
        return LightningModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
            resize_mode="standard",
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration={
                "input_names": ["images"],
                "output_names": ["bboxes", "labels", "scores"],
                "dynamic_shapes": {"inputs": {0: Dim("batch")}},
                "autograd_inlining": False,
                "opset_version": 18,
            },
            output_names=["bboxes", "labels", "scores"],
        )

    @property
    def _default_preprocessing_params(self) -> dict[str, DataInputParams]:  # type: ignore[override]
        """Default preprocessing parameters for RF-DETR models."""
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)

        return {
            "rfdetr_nano": DataInputParams(
                input_size=(384, 384),
                mean=imagenet_mean,
                std=imagenet_std,
            ),
            "rfdetr_small": DataInputParams(
                input_size=(512, 512),
                mean=imagenet_mean,
                std=imagenet_std,
            ),
            "rfdetr_base": DataInputParams(
                input_size=(560, 560),
                mean=imagenet_mean,
                std=imagenet_std,
            ),
            "rfdetr_medium": DataInputParams(
                input_size=(576, 576),
                mean=imagenet_mean,
                std=imagenet_std,
            ),
            "rfdetr_large": DataInputParams(
                input_size=(704, 704),
                mean=imagenet_mean,
                std=imagenet_std,
            ),
        }
