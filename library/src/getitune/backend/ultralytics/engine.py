# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Ultralytics engine implementation."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Sequence

import torch
from torchvision import tv_tensors

from getitune.data.entity.base import ImageInfo
from getitune.data.entity.sample import Prediction, SampleBatch
from getitune.data.module import DataModule
from getitune.engine.engine import Engine
from getitune.types.export import ExportFormat
from getitune.types.precision import Precision

from .models.base import UltralyticsModel

if TYPE_CHECKING:
    from getitune.types import PathLike
    from getitune.types.types import ANNOTATIONS, DATA, METRICS, MODEL

logger = logging.getLogger(__name__)


_ULTRALYTICS_FORMAT_MAP: dict[ExportFormat, str] = {
    ExportFormat.OPENVINO: "openvino",
    ExportFormat.ONNX: "onnx",
}


class _UltralyticsResultLike(Protocol):
    orig_img: Any
    orig_shape: tuple[int, int]


class UltralyticsEngine(Engine):
    """Engine backed by ``ultralytics.YOLO``.

    Wraps an :class:`UltralyticsModel` and a
    :class:`~getitune.data.module.DataModule` (or data-root path).
    """

    _EXPORTED_MODEL_BASE_NAME: ClassVar[str] = "exported_model"

    def __init__(
        self,
        model: UltralyticsModel,
        data: DataModule | PathLike,
        work_dir: PathLike = "./getitune-workspace",
        device: str = "auto",
        **kwargs,
    ) -> None:
        """Initialize the engine.

        Args:
            model: Ultralytics model wrapper.
            data: DataModule or filesystem data-root path.
            work_dir: Directory for checkpoints, exports, and logs.
            device: Device string (``"0"``, ``"cpu"``, ``"auto"``).
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

        if isinstance(data, DataModule):
            self._datamodule: DataModule | None = data
            self._data_root: Path | None = None
        elif isinstance(data, (str, os.PathLike)):
            self._datamodule = None
            self._data_root = Path(data)
        else:
            msg = f"data must be DataModule or PathLike, got {type(data)}"
            raise TypeError(msg)

    # ------------------------------------------------------------------
    # Engine interface
    # ------------------------------------------------------------------

    def train(self, **kwargs) -> METRICS:
        """Train the model via a custom Ultralytics trainer.

        Args:
            **kwargs: Overrides forwarded to Ultralytics training.

        Returns:
            Translated metric dict.
        """
        yolo = self._model.yolo
        merged = self._build_overrides(**kwargs)

        if self._data_root is not None and "data" not in merged:
            merged["data"] = str(self._data_root)

        trainer_cls = self._make_bound_trainer()

        logger.info(
            f"Starting Ultralytics training: model={self._model.model_name}, "
            f"device={self._device}, imgsz={self._model.imgsz}"
        )

        results = yolo.train(
            trainer=trainer_cls,
            device=self._device,
            imgsz=self._model.imgsz,
            project=str(self._work_dir),
            name="train",
            exist_ok=True,
            **merged,
        )
        return self._translate_metrics(results)

    def test(self, **kwargs) -> METRICS:
        """Validate the model.

        When a DataModule is attached, a custom validator bypasses the
        Ultralytics YAML data config and reads from the adapter pipeline.
        When a data-root path is attached, ``yolo.val()`` is used directly.

        Args:
            **kwargs: Overrides forwarded to validation.

        Returns:
            Translated metric dict.
        """
        merged = self._build_overrides(**kwargs)

        if self._datamodule is not None:
            return self._test_with_datamodule(merged)

        yolo = self._model.yolo
        if self._data_root is not None and "data" not in merged:
            merged["data"] = str(self._data_root)

        logger.info(f"Starting Ultralytics validation: model={self._model.model_name}")

        results = yolo.val(
            device=self._device,
            imgsz=self._model.imgsz,
            project=str(self._work_dir),
            name="val",
            exist_ok=True,
            **merged,
        )
        return self._translate_metrics(results)

    def predict(self, **kwargs) -> ANNOTATIONS:
        """Run inference and return a list of :class:`Prediction` objects.

        When a DataModule is attached, iterates ``predict_dataloader()``.
        Otherwise uses ``yolo.predict(source=...)``.

        Args:
            **kwargs: Overrides forwarded to prediction.
        """
        merged = self._build_overrides(**kwargs)

        if self._datamodule is not None and "source" not in merged:
            return self._predict_with_datamodule(merged)  # pyrefly: ignore[bad-return]

        yolo = self._model.yolo
        source: str | None = str(merged.pop("source")) if "source" in merged else None
        if source is None and self._data_root is not None:
            source = str(self._data_root)

        raw_results = yolo.predict(  # pyrefly: ignore[bad-argument-type]
            source=source,  # pyrefly: ignore[bad-argument-type]
            device=self._device,
            imgsz=self._model.imgsz,
            project=str(self._work_dir),
            name="predict",
            exist_ok=True,
            save=False,
            **merged,  # pyrefly: ignore[bad-argument-type]
        )

        return self._convert_predictions(raw_results)  # pyrefly: ignore[bad-return]

    def export(
        self,
        export_format: ExportFormat = ExportFormat.OPENVINO,
        export_precision: Precision = Precision.FP32,
        **kwargs,
    ) -> Path:
        """Export the model to OpenVINO IR or ONNX.

        Args:
            export_format: Target format.
            export_precision: Precision (FP32 or FP16).
            **kwargs: Extra arguments for Ultralytics export.

        Returns:
            Path to the exported model file.
        """
        yolo = self._model.yolo

        ultra_format = _ULTRALYTICS_FORMAT_MAP.get(export_format)
        if ultra_format is None:
            msg = f"Unsupported export format: {export_format}"
            raise ValueError(msg)

        half = export_precision == Precision.FP16

        logger.info(f"Exporting model: format={export_format.value}, precision={export_precision.value}")

        export_result = yolo.export(
            format=ultra_format,
            imgsz=self._model.imgsz,
            half=half,
            **kwargs,
        )
        export_path = Path(export_result)

        if export_format == ExportFormat.OPENVINO:
            return self._normalize_openvino_export(export_path)
        if export_format == ExportFormat.ONNX:
            return self._normalize_onnx_export(export_path)

        return export_path

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
    def datamodule(self) -> DATA:
        """The datamodule or data-root path."""
        if self._datamodule is not None:
            return self._datamodule  # type: ignore[return-value]
        if self._data_root is not None:
            return self._data_root  # type: ignore[return-value]
        msg = "No data source configured."
        raise RuntimeError(msg)

    # ------------------------------------------------------------------
    # DataModule-aware test / predict
    # ------------------------------------------------------------------

    def _test_with_datamodule(self, overrides: dict) -> dict[str, float]:
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    def _build_overrides(self, **kwargs) -> dict[str, Any]:
        """Merge overrides: model defaults < engine kwargs < call kwargs."""
        overrides: dict[str, Any] = {}
        overrides.update(self._model.extra_overrides)
        overrides.update(self._kwargs)
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

    def _normalize_openvino_export(self, export_path: Path) -> Path:
        """Copy OpenVINO export into work_dir and return the .xml path."""
        target_dir = self._work_dir / self._EXPORTED_MODEL_BASE_NAME

        if export_path.is_dir():
            if export_path != target_dir:
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.copytree(str(export_path), str(target_dir))
                logger.info(f"Exported model copied to {target_dir}")

            xml_files = list(target_dir.glob("*.xml"))
            if not xml_files:
                msg = f"No .xml file found in exported directory: {target_dir}"
                raise FileNotFoundError(msg)
            return xml_files[0]

        return export_path

    def _normalize_onnx_export(self, export_path: Path) -> Path:
        """Copy ONNX export into work_dir and return the .onnx path."""
        target_file = self._work_dir / f"{self._EXPORTED_MODEL_BASE_NAME}.onnx"

        if export_path != target_file:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(export_path), str(target_file))
            logger.info(f"Exported model copied to {target_file}")

        return target_file

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve ``"auto"`` to ``"0"`` (CUDA) or ``"cpu"``."""
        if device == "auto":
            return "0" if torch.cuda.is_available() else "cpu"
        return device
