# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Ultralytics engine implementation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Sequence

import torch
from torchvision import tv_tensors

from getitune.data.entity.base import ImageInfo
from getitune.data.entity.sample import Prediction, SampleBatch
from getitune.data.module import DataModule
from getitune.engine.engine import Engine
from getitune.types.device import DeviceType
from getitune.types.export import ExportFormat
from getitune.types.precision import Precision
from getitune.utils.device import is_xpu_available

from .models.base import UltralyticsModel

if TYPE_CHECKING:
    from collections.abc import Mapping

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

    def train(self, **kwargs) -> METRICS:
        """Train the model via a custom Ultralytics trainer.

        Args:
            **kwargs: Overrides forwarded to Ultralytics training.

        Returns:
            Translated metric dict.
        """
        yolo = self._model.yolo
        merged = self._build_overrides(self._train_args, **kwargs)

        if self._data_root is not None and "data" not in merged:
            merged["data"] = str(self._data_root)

        trainer_cls = self._make_bound_trainer()
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
        """Validate the model.

        When a DataModule is attached, a custom validator bypasses the
        Ultralytics YAML data config and reads from the adapter pipeline.
        When a data-root path is attached, ``yolo.val()`` is used directly.

        Args:
            checkpoint: Optional ``.pt`` checkpoint to validate.
            metric: Accepted for API compatibility with ``LightningEngine``
                but ignored — Ultralytics computes metrics internally.
            **kwargs: Overrides forwarded to validation.

        Returns:
            Translated metric dict.
        """
        if metric is not None:
            logger.debug("UltralyticsEngine ignores the 'metric' parameter; metrics are computed internally")
        merged = self._build_overrides(**kwargs)

        if self._datamodule is not None:
            return self._test_with_datamodule(merged, checkpoint=checkpoint)

        if checkpoint is not None:
            self._model.load_checkpoint(checkpoint)
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
        source: str | None = str(merged.pop("source")) if "source" in merged else None
        if source is None and self._data_root is not None:
            source = str(self._data_root)

        predict_args = {
            "source": source,
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
            checkpoint: Path to a ``.pt`` checkpoint to export.  When given,
                the model loads weights from this file before exporting.
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
        """The datamodule or data-root path."""
        if self._datamodule is not None:
            return self._datamodule  # type: ignore[return-value]
        if self._data_root is not None:
            return self._data_root  # type: ignore[return-value]
        msg = "No data source configured."
        raise RuntimeError(msg)

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

            imgs = self._batch_images_to_tensor(batch.images).to(device)
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

    def _make_bound_trainer(self) -> type:
        """Return a trainer subclass with the DataModule bound as a class attr."""
        base_cls = self._model.trainer_cls
        if base_cls is None:
            msg = f"{type(self._model).__name__} does not define a trainer_cls"
            raise TypeError(msg)

        if self._datamodule is None:
            return base_cls

        return type(base_cls.__name__, (base_cls,), {"_datamodule": self._datamodule})

    def _make_bound_validator(self) -> type:
        """Return a validator subclass with the DataModule bound as a class attr."""
        base_cls = self._model.validator_cls
        if base_cls is None:
            msg = f"{type(self._model).__name__} does not define a validator_cls"
            raise TypeError(msg)

        if self._datamodule is None:
            return base_cls

        return type(base_cls.__name__, (base_cls,), {"_datamodule": self._datamodule})

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
        """Map Ultralytics metric keys to getitune names."""
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

        return translated

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
    def _batch_images_to_tensor(
        images: torch.Tensor | tv_tensors.Image | list[torch.Tensor] | list[tv_tensors.Image],
    ) -> torch.Tensor:
        """Convert ``SampleBatch.images`` to a BCHW tensor."""
        if isinstance(images, torch.Tensor):
            return images
        return torch.stack([torch.as_tensor(image) for image in images], dim=0)

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

        img_tensor = torch.from_numpy(result.orig_img).permute(2, 0, 1).float() / 255.0
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
            return torch.device("cuda:0")

        # Bare integer index → CUDA device (Ultralytics convention).
        if device.isdigit():
            return torch.device(f"cuda:{device}")

        # Anything else (e.g. "cuda:1", "cpu") — let torch.device parse it.
        return torch.device(device)
