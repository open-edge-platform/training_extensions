# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Ultralytics engine implementation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Sequence

import torch
from torch.nn import functional
from torchvision import tv_tensors

from getitune.data.entity.base import ImageInfo
from getitune.data.entity.sample import Prediction, SampleBatch
from getitune.data.module import DataModule
from getitune.data.utils.structures.mask.mask_util import encode_rle
from getitune.engine.engine import Engine
from getitune.types.device import DeviceType
from getitune.types.export import ExportFormat
from getitune.types.precision import Precision
from getitune.utils.device import is_xpu_available

from .models.base import UltralyticsModel

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from ultralytics import YOLO

    from getitune.types import PathLike
    from getitune.types.types import ANNOTATIONS, DATA, METRICS, MODEL

logger = logging.getLogger(__name__)


class _UltralyticsResultLike(Protocol):
    orig_img: Any
    orig_shape: tuple[int, int]


class UltralyticsEngine(Engine):
    """Engine backed by ``ultralytics.YOLO``.

    Wraps an :class:`UltralyticsModel` and a
    :class:`~getitune.data.module.DataModule` (or data-root path).
    """

    _EXPORTED_MODEL_BASE_NAME: ClassVar[str] = "exported_model"
    _LAST_TRAIN_CHECKPOINT_FILE: ClassVar[str] = ".last_train_checkpoint"

    def __init__(
        self,
        model: UltralyticsModel,
        data: DataModule | PathLike,
        work_dir: PathLike = "./getitune-workspace",
        device: str | DeviceType = "auto",
        train_args: Mapping[str, Any] | None = None,
        export_args: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Initialize the engine.

        Args:
            model: Ultralytics model wrapper.
            data: DataModule or filesystem data-root path.
            work_dir: Directory for checkpoints, exports, and logs.
            device: Device string or :class:`DeviceType` enum
                (``"auto"``, ``"xpu"``, ``"0"``, ``"cpu"``, ``DeviceType.xpu``, etc.).
            train_args: Train-only defaults forwarded to ``yolo.train()``.
            export_args: Export metadata defaults, such as inference thresholds.
            **kwargs: Extra overrides forwarded to Ultralytics calls.
        """
        if not isinstance(model, UltralyticsModel):
            msg = f"model must be an UltralyticsModel instance, got {type(model)}"
            raise TypeError(msg)

        self._model = model
        self._work_dir = Path(work_dir).resolve()
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._device = self._resolve_device(device)
        self._kwargs = kwargs
        self._train_args = dict(train_args or {})
        self._export_args = dict(export_args or {})
        self._last_train_checkpoint = self._load_last_train_checkpoint()

        if isinstance(data, DataModule):
            self._datamodule: DataModule | None = data
            self._data_root: Path | None = None
        elif isinstance(data, (str, os.PathLike)):
            self._datamodule = None
            self._data_root = Path(data)
        else:
            msg = f"data must be DataModule or PathLike, got {type(data)}"
            raise TypeError(msg)

        # Propagate intensity config from DataModule so the exporter can embed
        # the correct input_dtype and intensity_mode into the exported model.
        if self._datamodule is not None:
            intensity_cfg = getattr(self._datamodule, "input_intensity_config", None)
            if intensity_cfg is not None:
                self._model.set_intensity_config(intensity_cfg)

    def train(
        self,
        epochs: int | None = None,
        batch: int | None = None,
        lr0: float | None = None,
        patience: int | None = None,
        max_epochs: int | None = None,
        callbacks: list | None = None,
        **kwargs,
    ) -> METRICS:
        """Train the model via a custom Ultralytics trainer.

        Args:
            epochs: Number of training epochs.
            batch: Batch size.
            lr0: Initial learning rate.
            patience: Early stopping patience (0 to disable).
            max_epochs: Alias for ``epochs`` (Lightning compatibility).
            callbacks: Accepted for API compatibility; unused by Ultralytics.
            **kwargs: Additional overrides forwarded to Ultralytics training.

        Returns:
            Translated metric dict.
        """
        progress_fn, progress_min, progress_max = self._extract_progress_callback(callbacks)
        # Translate Lightning-specific 'devices' to Ultralytics device selection.
        if "devices" in kwargs:
            devices = kwargs.pop("devices")
            if isinstance(devices, list) and len(devices) >= 1:
                idx = devices[0]
                if len(devices) > 1:
                    logger.warning(f"UltralyticsEngine does not support multi-device; using first device: {idx}")
                # Only set index for accelerator devices; CPU doesn't support indices.
                dev_type = self._device.type
                if dev_type != "cpu":
                    self._device = torch.device(f"{dev_type}:{idx}")
        # Drop Lightning-only kwargs that have no Ultralytics equivalent.
        kwargs.pop("precision", None)
        explicit: dict[str, Any] = {}
        if epochs is not None:
            explicit["epochs"] = epochs
        elif max_epochs is not None:
            explicit["epochs"] = max_epochs
        if batch is not None:
            explicit["batch"] = batch
        if lr0 is not None:
            explicit["lr0"] = lr0
        if patience is not None:
            explicit["patience"] = patience
        kwargs.update(explicit)
        yolo = self._model.yolo
        merged = self._build_overrides(self._train_args, **kwargs)

        if self._data_root is not None and "data" not in merged:
            merged["data"] = str(self._data_root)

        trainer_cls = self._make_bound_trainer(
            progress_fn=progress_fn,
            progress_min=progress_min,
            progress_max=progress_max,
        )
        train_args = {
            "trainer": trainer_cls,
            "device": self._device,
            "imgsz": self._model.imgsz,
            "project": str(self._work_dir),
            "name": "train",
            "exist_ok": True,
            **merged,
        }

        logger.info(
            f"Starting Ultralytics training: model={self._model.model_name}, "
            f"device={self._device}, imgsz={self._model.imgsz}"
        )

        results = yolo.train(**train_args)
        self._record_last_train_checkpoint(self._resolve_trainer_checkpoint(yolo))
        return self._translate_metrics(results)

    def test(self, checkpoint: PathLike | None = None, metric: object | None = None, **kwargs) -> METRICS:
        """Evaluate the model using torchmetrics or the Ultralytics validator.

        When a ``metric`` callable is provided **and** a DataModule is
        attached, evaluation uses the same torchmetrics pipeline as
        Lightning (e.g. ``MeanAveragePrecision``).  This ensures metric
        consistency across backends.

        When ``metric`` is ``None``, falls back to the Ultralytics
        built-in validator (``DetMetrics``).

        Args:
            checkpoint: Optional ``.pt`` checkpoint to evaluate.
            metric: A ``MetricCallable`` — a function that accepts
                ``LabelInfo`` and returns a ``torchmetrics.Metric``.
                When provided, the torchmetrics evaluation path is used.
            **kwargs: Overrides forwarded to validation.

        Returns:
            Metric dict.  Keys are prefixed with ``test/`` when using
            torchmetrics, or ``val/`` when using the YOLO validator.
        """
        if checkpoint is not None:
            self._model.load_checkpoint(checkpoint)

        if metric is not None and self._datamodule is not None:
            return self._test_with_torchmetrics(metric)

        merged = self._build_overrides(**kwargs)

        if self._datamodule is not None:
            return self._test_with_datamodule(merged, checkpoint=None)

        yolo = self._model.yolo
        if self._data_root is not None and "data" not in merged:
            merged["data"] = str(self._data_root)

        val_args = {
            "device": self._device,
            "imgsz": self._model.imgsz,
            "project": str(self._work_dir),
            "name": "val",
            "exist_ok": True,
            **merged,
        }

        logger.info(f"Starting Ultralytics validation: model={self._model.model_name}")

        results = yolo.val(**val_args)
        return self._translate_metrics(results)

    def predict(
        self, source: str | Path | None = None, conf: float | None = None, iou: float | None = None, **kwargs
    ) -> ANNOTATIONS:
        """Run inference and return a list of :class:`Prediction` objects.

        When a DataModule is attached, iterates ``predict_dataloader()``.
        Otherwise uses ``yolo.predict(source=...)``.

        Args:
            source: Image source path or directory. Overrides attached data.
            conf: Confidence threshold for predictions.
            iou: IoU threshold for NMS.
            **kwargs: Additional overrides forwarded to prediction.
        """
        extra: dict[str, Any] = {}
        if source is not None:
            extra["source"] = str(source)
        if conf is not None:
            extra["conf"] = conf
        if iou is not None:
            extra["iou"] = iou
        merged = self._build_overrides(**extra, **kwargs)

        if self._datamodule is not None and "source" not in merged:
            return self._predict_with_datamodule(merged)  # pyrefly: ignore[bad-return]

        yolo = self._model.yolo
        resolved_source = str(merged.pop("source")) if "source" in merged else None
        if resolved_source is None and self._data_root is not None:
            resolved_source = str(self._data_root)

        predict_args = {
            "source": resolved_source,
            "device": self._device,
            "imgsz": self._model.imgsz,
            "project": str(self._work_dir),
            "name": "predict",
            "exist_ok": True,
            "save": False,
            **merged,
        }

        raw_results = yolo.predict(**predict_args)  # pyrefly: ignore[bad-argument-type]

        return self._convert_predictions(raw_results)  # pyrefly: ignore[bad-return]

    def export(
        self,
        checkpoint: PathLike | None = None,
        export_format: ExportFormat = ExportFormat.OPENVINO,
        export_precision: Precision = Precision.FP32,
        **kwargs,
    ) -> Path:
        """Export the model to OpenVINO IR or ONNX.

        Delegates to :meth:`UltralyticsModel.export` which follows the same
        architecture as Lightning — metadata embedding, preprocessing
        parameters, and FP16 compression are handled by the model's exporter.

        Args:
            checkpoint: Path to weights to export. When given, the model loads
                weights from this file before exporting.
            export_format: Target format.
            export_precision: Precision (FP32 or FP16).
            **kwargs: Extra arguments (reserved for future use).

        Returns:
            Path to the exported model file (``.xml`` for OpenVINO,
            ``.onnx`` for ONNX).
        """
        if checkpoint is not None:
            self._model.load_checkpoint(checkpoint)
        elif self._last_train_checkpoint is not None and self._last_train_checkpoint.exists():
            self._model.load_checkpoint(self._last_train_checkpoint)
        else:
            best_pt = self._work_dir / "train" / "weights" / "best.pt"
            if best_pt.exists():
                self._model.load_checkpoint(best_pt)

        logger.info(
            f"Exporting model: format={export_format.value}, "
            f"precision={export_precision.value}, "
            f"checkpoint={checkpoint or self._last_train_checkpoint or 'current weights'}"
        )

        return self._model.export(
            output_dir=self._work_dir,
            base_name=self._EXPORTED_MODEL_BASE_NAME,
            export_format=export_format,
            precision=export_precision,
            export_args=self._export_args,
        )

    @staticmethod
    def is_supported(model: MODEL, data: DATA) -> bool:
        """Return ``True`` when *model* is an :class:`UltralyticsModel`."""
        return bool(isinstance(model, UltralyticsModel) and isinstance(data, (DataModule, str, os.PathLike)))

    @property
    def work_dir(self) -> PathLike:
        """Working directory."""
        return self._work_dir

    @property
    def model(self) -> UltralyticsModel:
        """The wrapped :class:`UltralyticsModel`."""
        return self._model  # type: ignore[return-value]

    @property
    def best_checkpoint(self) -> Path | None:
        """Path to the best model checkpoint after training.

        Resolution order:
        1. Recorded checkpoint from the most recent ``train()`` call.
        2. ``best.pt`` from the default training run directory.
        3. ``None`` if no checkpoint is available.
        """
        if self._last_train_checkpoint is not None and self._last_train_checkpoint.exists():
            return self._last_train_checkpoint
        default_best = self._work_dir / "train" / "weights" / "best.pt"
        return default_best if default_best.exists() else None

    @property
    def datamodule(self) -> DATA:
        """The attached DataModule, or ``None``."""
        if self._datamodule is not None:
            return self._datamodule  # type: ignore[return-value]
        return None  # type: ignore[return-value]

    def _test_with_datamodule(self, overrides: dict, checkpoint: PathLike | None = None) -> dict[str, float]:
        """Run validation via a bound validator class with DataModule data."""
        validator_cls = self._make_bound_validator()

        args = {
            "imgsz": self._model.imgsz,
            "device": self._device,
            "project": str(self._work_dir),
            "name": "val",
            "exist_ok": True,
            **overrides,
            "mode": "val",
        }

        logger.info(f"Starting DataModule validation: model={self._model.model_name}")

        validator = validator_cls(
            save_dir=self._work_dir / "val",
            args=args,
        )

        if checkpoint is not None:
            self._model.load_checkpoint(checkpoint)
        results = validator(model=self._model.yolo.model)
        return self._translate_metrics(results)

    def _test_with_torchmetrics(self, metric_callable: Any) -> dict[str, float]:  # noqa: ANN401
        """Evaluate using torchmetrics — same metrics as Lightning models.

        Iterates the DataModule's test dataloader, runs YOLO predictions on
        each batch, converts outputs to the torchmetrics dict format, and
        computes the metric.  Returns a flat dict with ``test/`` prefixed keys
        identical to those produced by ``LightningEngine.test()``.

        Supports both detection and instance segmentation:

        - **Detection**: predictions contain ``boxes``, ``scores``, ``labels``.
        - **Instance segmentation**: predictions additionally contain ``masks``
          as RLE-encoded dicts, matching ``MaskRLEMeanAveragePrecision`` format.

        Target bounding boxes and masks from the DataModule are in original
        image coordinates (``resize_targets=False``), while YOLO predictions
        are in the letterbox-padded model input space.  This method transforms
        both boxes and masks into the prediction coordinate space before
        metric update.

        Args:
            metric_callable: A function ``(LabelInfo) -> Metric``.

        Returns:
            Flat metric dict, e.g. ``{"test/map": 0.75, "test/map_50": 0.90}``.
        """
        assert self._datamodule is not None  # noqa: S101

        label_info = self._model.label_info or self._datamodule.label_info
        metric = metric_callable(label_info)
        device = self._device

        yolo = self._model.yolo
        yolo.model.to(device).eval()  # pyrefly: ignore[missing-attribute]
        metric = metric.to(device)

        dataloader = self._datamodule.test_dataloader()
        imgsz = self._model.imgsz

        logger.info(
            f"Starting torchmetrics evaluation: model={self._model.model_name}, "
            f"metric={type(metric).__name__}, batches={len(dataloader)}"
        )

        for batch in dataloader:
            if not isinstance(batch, SampleBatch):
                msg = f"Expected test_dataloader to yield SampleBatch, got {type(batch)}"
                raise TypeError(msg)

            imgs = batch.images.to(device) if isinstance(batch.images, torch.Tensor) else batch.images
            raw_results = yolo.predict(  # pyrefly: ignore[bad-argument-type]
                source=imgs,  # pyrefly: ignore[bad-argument-type]
                device=device,
                imgsz=imgsz,
                conf=0.0,
                save=False,
                verbose=False,
            )

            preds_list = []
            for result in raw_results:
                pred_dict: dict[str, Any] = {
                    "boxes": torch.zeros((0, 4), device=device),
                    "scores": torch.zeros(0, device=device),
                    "labels": torch.zeros(0, dtype=torch.long, device=device),
                }
                if result.boxes is not None and len(result.boxes):
                    pred_dict["boxes"] = result.boxes.xyxy.to(device)  # pyrefly: ignore[missing-attribute]
                    pred_dict["scores"] = result.boxes.conf.to(device).float()  # pyrefly: ignore[missing-attribute]
                    pred_dict["labels"] = result.boxes.cls.to(device).long()  # pyrefly: ignore[missing-attribute]
                if result.masks is not None and len(result.masks):
                    masks_data = result.masks.data  # pyrefly: ignore[missing-attribute]
                    pred_dict["masks"] = [encode_rle((m > 0.5).cpu()) for m in masks_data]
                preds_list.append(pred_dict)

            target_list = []
            for i in range(len(raw_results)):
                tgt_dict: dict[str, Any] = {
                    "boxes": torch.zeros((0, 4), device=device),
                    "labels": torch.zeros(0, dtype=torch.long, device=device),
                }
                if batch.bboxes is not None and i < len(batch.bboxes):
                    boxes = batch.bboxes[i].data.to(device).float()
                    ori_h, ori_w = batch.bboxes[i].canvas_size
                    boxes = self._scale_boxes_to_letterbox(boxes, ori_h, ori_w, imgsz)
                    tgt_dict["boxes"] = boxes
                if batch.labels is not None and i < len(batch.labels):
                    tgt_dict["labels"] = batch.labels[i].to(device).long()
                if batch.masks is not None and i < len(batch.masks):
                    target_masks = batch.masks[i].data  # (N, ori_h, ori_w)
                    mask_ori_h, mask_ori_w = target_masks.shape[-2:]
                    scaled_masks = self._scale_masks_to_letterbox(target_masks, mask_ori_h, mask_ori_w, imgsz)
                    tgt_dict["masks"] = [encode_rle(m) for m in scaled_masks]
                target_list.append(tgt_dict)

            metric.update(preds=preds_list, target=target_list)

        results = metric.compute()
        return self._format_torchmetrics_results(results)

    @staticmethod
    def _scale_boxes_to_letterbox(boxes: torch.Tensor, ori_h: int, ori_w: int, imgsz: int) -> torch.Tensor:
        """Transform bounding boxes from original image coords to letterbox-padded coords.

        The DataModule's test augmentations use ``Resize(keep_aspect_ratio=True,
        center_padding=True, resize_targets=False)``, so target boxes remain in
        the original image coordinate space.  YOLO predictions, however, are in
        the letterbox-padded ``imgsz x imgsz`` space.  This method applies the
        same scale + center-pad offset to align the two.

        Args:
            boxes: ``(N, 4)`` tensor of xyxy boxes in original coords.
            ori_h: Original image height.
            ori_w: Original image width.
            imgsz: Model input size (square).

        Returns:
            Transformed ``(N, 4)`` tensor in letterbox coords.
        """
        if boxes.numel() == 0:
            return boxes
        scale = min(imgsz / ori_h, imgsz / ori_w)
        pad_x = (imgsz - ori_w * scale) / 2.0
        pad_y = (imgsz - ori_h * scale) / 2.0
        scaled = boxes.clone()
        scaled[:, 0] = boxes[:, 0] * scale + pad_x
        scaled[:, 1] = boxes[:, 1] * scale + pad_y
        scaled[:, 2] = boxes[:, 2] * scale + pad_x
        scaled[:, 3] = boxes[:, 3] * scale + pad_y
        return scaled

    @staticmethod
    def _scale_masks_to_letterbox(masks: torch.Tensor, ori_h: int, ori_w: int, imgsz: int) -> torch.Tensor:
        """Transform binary masks from original image coords to letterbox-padded coords.

        Analogous to ``_scale_boxes_to_letterbox`` but for spatial mask tensors.
        Resizes each mask with nearest-neighbor interpolation to preserve binary
        values, then center-pads to ``(imgsz, imgsz)``.

        Args:
            masks: ``(N, ori_h, ori_w)`` binary mask tensor.
            ori_h: Original image height.
            ori_w: Original image width.
            imgsz: Model input size (square).

        Returns:
            ``(N, imgsz, imgsz)`` binary mask tensor in letterbox coords.
        """
        if masks.numel() == 0:
            return torch.zeros((0, imgsz, imgsz), dtype=torch.bool, device=masks.device)

        scale = min(imgsz / ori_h, imgsz / ori_w)
        new_h = round(ori_h * scale)
        new_w = round(ori_w * scale)

        resized = functional.interpolate(
            masks.unsqueeze(1).float(),
            size=(new_h, new_w),
            mode="nearest",
        ).squeeze(1)

        pad_y = (imgsz - new_h) // 2
        pad_x = (imgsz - new_w) // 2

        result = torch.zeros((masks.shape[0], imgsz, imgsz), dtype=torch.bool, device=masks.device)
        result[:, pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized > 0.5
        return result

    def _predict_with_datamodule(self, overrides: dict[str, Any]) -> list[Prediction]:
        """Run inference through ``DataModule.predict_dataloader()``."""
        assert self._datamodule is not None  # guaranteed by caller  # noqa: S101
        overrides.pop("batch", None)
        dataloader = self._datamodule.predict_dataloader()

        yolo = self._model.yolo
        device = self._device
        yolo.model.to(device).eval()  # pyrefly: ignore[missing-attribute]

        predictions: list[Prediction] = []
        for batch in dataloader:
            if not isinstance(batch, SampleBatch):
                msg = f"Expected DataModule.predict_dataloader() to yield SampleBatch, got {type(batch)}"
                raise TypeError(msg)
            if not isinstance(batch.images, torch.Tensor):
                msg = f"Expected collated SampleBatch.images to be a tensor, got {type(batch.images)}"
                raise TypeError(msg)

            imgs = batch.images.to(device)
            raw_results = yolo.predict(  # pyrefly: ignore[bad-argument-type]
                source=imgs,  # pyrefly: ignore[bad-argument-type]
                device=device,
                imgsz=self._model.imgsz,
                save=False,
                verbose=False,
                **overrides,
            )
            predictions.extend(self._convert_predictions(raw_results, images=batch.images, imgs_info=batch.imgs_info))

        return predictions

    def _resolve_trainer_checkpoint(self, yolo: YOLO) -> Path | None:
        """Return the actual checkpoint produced by the latest training run."""
        trainer = getattr(yolo, "trainer", None)
        if trainer is None:
            return None

        for attr in ("best", "last"):
            checkpoint = getattr(trainer, attr, None)
            if checkpoint is None:
                continue
            checkpoint_path = Path(checkpoint).resolve()
            if checkpoint_path.exists():
                return checkpoint_path

        return None

    def _load_last_train_checkpoint(self) -> Path | None:
        """Load the persisted latest-training checkpoint pointer, if present."""
        checkpoint_file = self._work_dir / self._LAST_TRAIN_CHECKPOINT_FILE
        if not checkpoint_file.exists():
            return None

        checkpoint_text = checkpoint_file.read_text(encoding="utf-8").strip()
        if not checkpoint_text:
            return None

        checkpoint = Path(checkpoint_text).resolve()
        if checkpoint.exists():
            return checkpoint

        logger.warning(f"Ignoring stale checkpoint pointer: {checkpoint}")
        return None

    def _record_last_train_checkpoint(self, checkpoint: Path | None) -> None:
        """Persist the latest training checkpoint for later export resolution."""
        checkpoint_file = self._work_dir / self._LAST_TRAIN_CHECKPOINT_FILE
        if checkpoint is None:
            self._last_train_checkpoint = None
            if checkpoint_file.exists():
                checkpoint_file.unlink()
            return

        resolved_checkpoint = checkpoint.resolve()
        self._last_train_checkpoint = resolved_checkpoint
        checkpoint_file.write_text(str(resolved_checkpoint), encoding="utf-8")

    @staticmethod
    def _create_datamodule(data_root: Path, model: UltralyticsModel) -> DataModule:
        """Create a DataModule from a data-root path via AutoConfigurator.

        Maps the Ultralytics task string (``"detect"``, ``"segment"``) to a
        :class:`TaskType` so that ``AutoConfigurator`` can build the correct
        dataset pipeline.
        """
        from getitune.tools.auto_configurator import AutoConfigurator
        from getitune.types.task import TaskType

        _task_map: dict[str, TaskType] = {
            "detect": TaskType.DETECTION,
            "segment": TaskType.INSTANCE_SEGMENTATION,
        }
        task = _task_map.get(model.task)
        if task is None:
            msg = f"Cannot create DataModule for Ultralytics task '{model.task}'"
            raise ValueError(msg)

        auto_cfg = AutoConfigurator(data_root=data_root, task=task)
        return auto_cfg.get_datamodule()

    def _make_bound_trainer(
        self,
        progress_fn: Callable[[float], None] | None = None,
        progress_min: float = 0.0,
        progress_max: float = 100.0,
    ) -> type:
        """Return a trainer subclass with the DataModule bound as a class attr."""
        base_cls = self._model.trainer_cls
        if base_cls is None:
            msg = f"{type(self._model).__name__} does not define a trainer_cls"
            raise TypeError(msg)

        if self._datamodule is None:
            return base_cls

        attrs: dict[str, Any] = {
            "_datamodule": self._datamodule,
            "_use_getitune_data": True,
            "_progress_fn": progress_fn,
            "_progress_min": progress_min,
            "_progress_max": progress_max,
        }
        return type(base_cls.__name__, (base_cls,), attrs)

    @staticmethod
    def _extract_progress_callback(
        callbacks: list | None,
    ) -> tuple[Callable[[float], None] | None, float, float]:
        """Extract progress reporting callable from Lightning-style callbacks.

        Scans for any callback with ``_on_progress_update``, ``_min_p``, and
        ``_max_p`` attributes (duck-typed to avoid coupling to the application
        backend's ``TrainingProgressCallback``).

        Returns:
            (progress_fn, min_p, max_p) or (None, 0, 100) if not found.
        """
        if callbacks is None:
            return None, 0.0, 100.0

        for cb in callbacks:
            fn = getattr(cb, "_on_progress_update", None)
            if fn is not None:
                min_p = getattr(cb, "_min_p", 0.0)
                max_p = getattr(cb, "_max_p", 100.0)
                return fn, min_p, max_p

        return None, 0.0, 100.0

    def _make_bound_validator(self) -> type:
        """Return a validator subclass with the DataModule bound as a class attr."""
        base_cls = self._model.validator_cls
        if base_cls is None:
            msg = f"{type(self._model).__name__} does not define a validator_cls"
            raise TypeError(msg)

        if self._datamodule is None:
            return base_cls

        return type(base_cls.__name__, (base_cls,), {"_datamodule": self._datamodule, "_use_getitune_data": True})

    def _build_overrides(self, defaults: Mapping[str, Any] | None = None, **kwargs) -> dict[str, Any]:
        """Merge overrides: model defaults < engine kwargs < defaults < call kwargs."""
        overrides: dict[str, Any] = {}
        overrides.update(self._model.extra_overrides)
        overrides.update(self._kwargs)
        if defaults is not None:
            overrides.update(defaults)
        overrides.update(kwargs)
        return overrides

    def _translate_metrics(self, results: object) -> dict[str, float]:
        """Map Ultralytics metric keys to getitune names.

        Translates aggregate metrics via ``metric_keys`` and extracts
        per-class precision, recall, mAP50, and mAP50-95 when available.
        Per-class keys use the format ``val/<metric>/<class_name>``.
        """
        if results is None:
            return {}

        raw_metrics: dict[str, float] = {}
        if hasattr(results, "results_dict"):
            raw_metrics = dict(results.results_dict)
        elif isinstance(results, dict):
            raw_metrics = dict(results)

        translated: dict[str, float] = {}
        for ultra_key, value in raw_metrics.items():
            gt_key = self._model.metric_keys.get(ultra_key, f"ultralytics/{ultra_key}")
            translated[gt_key] = float(value) if not isinstance(value, float) else value

        self._add_per_class_metrics(results, translated)
        return translated

    @staticmethod
    def _add_per_class_metrics(results: object, metrics: dict[str, float]) -> None:
        """Extract per-class metrics from the Ultralytics results object.

        Adds keys like ``val/precision/<ClassName>``, ``val/recall/<ClassName>``,
        ``val/map_50/<ClassName>``, ``val/map/<ClassName>`` to *metrics* in-place.
        """
        names = getattr(results, "names", None)
        ap_class_index = getattr(results, "ap_class_index", None)
        if names is None or ap_class_index is None:
            return

        for i, cls_idx in enumerate(ap_class_index):
            class_name = names.get(cls_idx, str(cls_idx))
            try:
                result = results.class_result(i)  # pyrefly: ignore[missing-attribute]
            except (IndexError, AttributeError, TypeError):
                continue
            # Detection returns 4 values (p, r, ap50, ap)
            # Segmentation returns 8 values (box p, r, ap50, ap, mask p, r, ap50, ap)
            if len(result) >= 4:
                p, r, ap50, ap = result[:4]
                metrics[f"val/precision/{class_name}"] = float(p)
                metrics[f"val/recall/{class_name}"] = float(r)
                metrics[f"val/map_50/{class_name}"] = float(ap50)
                metrics[f"val/map/{class_name}"] = float(ap)
            if len(result) >= 8:
                mp, mr, map50, map_ = result[4:8]
                metrics[f"val/mask_precision/{class_name}"] = float(mp)
                metrics[f"val/mask_recall/{class_name}"] = float(mr)
                metrics[f"val/mask_map_50/{class_name}"] = float(map50)
                metrics[f"val/mask_map/{class_name}"] = float(map_)

    @staticmethod
    def _format_torchmetrics_results(results: dict[str, Any]) -> dict[str, float]:
        """Convert torchmetrics compute output to a flat ``test/``-prefixed dict.

        Mirrors the logic in ``LightningModel._log_metrics``: only scalar
        tensors are included; auxiliary keys (``classes``, ``map_per_class``,
        ``mar_100_per_class``, ``ious``) are skipped.

        Args:
            results: Dict returned by ``metric.compute()``.

        Returns:
            Flat dict, e.g. ``{"test/map": 0.75, "test/map_50": 0.90}``.
        """
        _skip_keys = {"classes", "map_per_class", "mar_100_per_class", "ious"}
        formatted: dict[str, float] = {}
        for name, value in results.items():
            if name in _skip_keys:
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    formatted[f"test/{name}"] = value.item()
                else:
                    logger.debug(f"Skipping non-scalar torchmetric '{name}' with {value.numel()} elements")
            elif isinstance(value, (int, float)):
                formatted[f"test/{name}"] = float(value)
        return formatted

    @staticmethod
    def _convert_predictions(
        raw_results: list[Any],
        images: torch.Tensor | tv_tensors.Image | list[torch.Tensor] | list[tv_tensors.Image] | None = None,
        imgs_info: Sequence[ImageInfo | None] | None = None,
    ) -> list[Prediction]:
        """Convert Ultralytics ``Results`` to getitune ``Prediction``."""
        predictions: list[Prediction] = []
        for idx, result in enumerate(raw_results):
            img_tensor, img_info = UltralyticsEngine._resolve_prediction_input(idx, result, images, imgs_info)
            h, w = img_info.ori_shape

            bboxes = None
            scores = None
            labels = None
            masks = None

            if result.boxes is not None and len(result.boxes):
                boxes_xyxy = torch.as_tensor(result.boxes.xyxy).cpu()  # pyrefly: ignore[missing-attribute]
                bboxes = tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
                    boxes_xyxy,
                    format=tv_tensors.BoundingBoxFormat.XYXY,
                    canvas_size=(h, w),
                )
                scores = torch.as_tensor(result.boxes.conf).cpu()  # pyrefly: ignore[missing-attribute]
                labels = torch.as_tensor(result.boxes.cls).cpu().long()  # pyrefly: ignore[missing-attribute]

            if result.masks is not None and len(result.masks):
                masks = tv_tensors.Mask(torch.as_tensor(result.masks.data).cpu())  # pyrefly: ignore[missing-attribute]

            predictions.append(
                Prediction(
                    image=tv_tensors.Image(img_tensor),
                    img_info=img_info,
                    bboxes=bboxes,
                    scores=scores,
                    label=labels,
                    masks=masks,
                ),
            )

        return predictions

    @staticmethod
    def _resolve_prediction_input(
        idx: int,
        result: _UltralyticsResultLike,
        images: torch.Tensor | tv_tensors.Image | list[torch.Tensor] | list[tv_tensors.Image] | None,
        imgs_info: Sequence[ImageInfo | None] | None,
    ) -> tuple[torch.Tensor, ImageInfo]:
        """Use batch inputs when available, otherwise fall back to Ultralytics results."""
        if images is not None:
            image = images[idx]
            img_tensor = torch.as_tensor(image).detach().cpu().float()
            img_info = imgs_info[idx] if imgs_info is not None else None
            if img_info is None:
                _, h, w = img_tensor.shape
                img_info = ImageInfo(  # pyrefly: ignore[no-matching-overload]
                    img_idx=idx,
                    img_shape=(h, w),
                    ori_shape=(h, w),
                )
            return img_tensor, img_info

        img_tensor = torch.from_numpy(result.orig_img).permute(2, 0, 1).float()
        h, w = result.orig_shape[0], result.orig_shape[1]
        img_info = ImageInfo(  # pyrefly: ignore[no-matching-overload]
            img_idx=idx,
            img_shape=(h, w),
            ori_shape=(h, w),
        )
        return img_tensor, img_info

    @staticmethod
    def _resolve_device(device: str | DeviceType) -> torch.device:
        """Resolve a device specification to a :class:`torch.device`.

        Resolution order for ``"auto"`` / ``DeviceType.auto``:
        XPU > CUDA > CPU (matches getitune convention).

        We return a ``torch.device`` object rather than a plain string so that
        Ultralytics' ``select_device()`` passes it through unchanged — its
        validator only rejects *string* device names it doesn't recognise,
        but accepts pre-constructed ``torch.device`` objects verbatim.

        Args:
            device: Raw string (``"auto"``, ``"xpu"``, ``"xpu:0"``, ``"cuda"``,
                ``"cuda:0"``, ``"0"``, ``"cpu"``) or :class:`DeviceType` enum.

        Returns:
            Resolved ``torch.device``.
        """
        # Normalise DeviceType enum to string.
        if isinstance(device, DeviceType):
            device = {
                DeviceType.auto: "auto",
                DeviceType.xpu: "xpu",
                DeviceType.gpu: "cuda",
                DeviceType.cpu: "cpu",
            }.get(device, str(device.value))

        device = str(device).strip().lower()

        if device == "auto":
            if is_xpu_available():
                return torch.device("xpu")
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

        if device == "xpu":
            return torch.device("xpu")

        if device in ("cuda", "gpu"):
            return torch.device("cuda")

        # Bare integer index → CUDA device (Ultralytics convention).
        if device.isdigit():
            return torch.device(f"cuda:{device}")

        # Anything else (e.g. "cuda:1", "cpu") — let torch.device parse it.
        return torch.device(device)
