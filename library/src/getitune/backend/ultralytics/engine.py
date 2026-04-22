# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Ultralytics engine implementation."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import torch
from torchvision import tv_tensors

from getitune.data.entity.base import ImageInfo
from getitune.data.entity.sample import Prediction
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


class UltralyticsEngine(Engine):
    """Engine backed by ``ultralytics.YOLO``.

    Wraps an :class:`UltralyticsModel` and a getitune
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
            model: :class:`UltralyticsModel` (detection or instance-seg).
            data: :class:`~getitune.data.module.DataModule` or data-root path.
            work_dir: Directory for checkpoints, exports, and logs.
            device: Ultralytics device string (``"0"``, ``"cpu"``, ``"auto"``).
            **kwargs: Extra overrides forwarded to Ultralytics calls.
        """
        if not isinstance(model, UltralyticsModel):
            msg = f"model must be an UltralyticsModel instance, got {type(model)}"
            raise TypeError(msg)

        self._model = model
        self._work_dir = Path(work_dir)
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

        The engine instantiates the model's ``trainer_cls`` (a custom
        Ultralytics trainer subclass) and routes training through it.
        When a :class:`~getitune.data.module.DataModule` is attached,
        the trainer receives a reference to it; Phase 2 overrides will
        use it for data loading.  When a filesystem path is attached,
        the default Ultralytics data loading is used.

        Args:
            **kwargs: Overrides forwarded to Ultralytics training.

        Returns:
            Translated metric dict.
        """
        yolo = self._model.yolo
        merged = self._build_overrides(**kwargs)

        # Resolve data source for Ultralytics
        if self._data_root is not None and "data" not in merged:
            merged["data"] = str(self._data_root)

        trainer_cls = self._make_bound_trainer()

        model_name = self._model.model_name
        logger.info(
            f"Starting Ultralytics training: model={model_name}, device={self._device}, imgsz={self._model.imgsz}"
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

        Uses ``yolo.val()`` under the hood.  When a data-root path is
        attached, it is forwarded as the ``data`` argument.  DataModule-based
        validation will be wired in Phase 2 (custom validator bridge).

        Args:
            **kwargs: Overrides forwarded to ``yolo.val()``.

        Returns:
            Translated metric dict.
        """
        yolo = self._model.yolo
        merged = self._build_overrides(**kwargs)

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

        Currently uses ``yolo.predict(source=...)``.  DataModule-based
        prediction (iterating ``DataModule.predict_dataloader()``) will
        be wired in Phase 2.

        Args:
            **kwargs: Overrides forwarded to ``yolo.predict()``.
        """
        yolo = self._model.yolo
        merged = self._build_overrides(**kwargs)

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
        )

        return self._convert_predictions(raw_results)  # pyrefly: ignore[bad-return]

    def export(
        self,
        export_format: ExportFormat = ExportFormat.OPENVINO,
        export_precision: Precision = Precision.FP32,
        **kwargs,
    ) -> Path:
        """Export the model to OpenVINO IR or ONNX.

        The exported artefacts are normalised into ``<work_dir>/`` and a
        concrete file path is returned (not a directory).

        Phase 4 will add full normalization; Phase 8 will add metadata
        embedding for OVEngine / ModelAPI compatibility.

        Args:
            export_format: Target format (``OPENVINO`` or ``ONNX``).
            export_precision: Precision (``FP32`` or ``FP16``).
            **kwargs: Extra arguments for Ultralytics export.

        Returns:
            Path to the exported model file (``.xml`` for OpenVINO,
            ``.onnx`` for ONNX).
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_bound_trainer(self) -> type:
        """Return a trainer class with the current DataModule bound.

        Creates a dynamic subclass of ``self._model.trainer_cls`` with the
        DataModule set as a class attribute.  Ultralytics instantiates the
        trainer class internally (fixed constructor signature), so this is
        the way to inject external state.
        """
        base_cls = self._model.trainer_cls
        if base_cls is None:
            msg = f"{type(self._model).__name__} does not define a trainer_cls"
            raise TypeError(msg)

        if self._datamodule is None:
            return base_cls

        # Dynamic subclass with DataModule baked in as a class variable.
        return type(base_cls.__name__, (base_cls,), {"_datamodule": self._datamodule})

    def _build_overrides(self, **kwargs) -> dict[str, object]:
        """Merge overrides: model defaults < engine kwargs < call kwargs."""
        overrides: dict[str, object] = {}
        overrides.update(self._model.extra_overrides)
        overrides.update(self._kwargs)
        overrides.update(kwargs)
        return overrides

    def _translate_metrics(self, results: object) -> dict[str, float]:
        """Map Ultralytics metric keys to getitune names.

        Unknown keys are kept with an ``ultralytics/`` prefix.
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

        return translated

    @staticmethod
    def _convert_predictions(raw_results: list) -> list[Prediction]:
        """Convert Ultralytics ``Results`` objects to getitune ``Prediction``."""
        predictions: list[Prediction] = []
        for idx, result in enumerate(raw_results):
            img_tensor = torch.from_numpy(result.orig_img).permute(2, 0, 1).float() / 255.0
            h, w = result.orig_shape[0], result.orig_shape[1]
            img_info = ImageInfo(  # pyrefly: ignore[no-matching-overload]
                img_idx=idx,
                img_shape=(h, w),
                ori_shape=(h, w),
            )

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

        # Ultralytics returned a file path directly (unlikely for OV).
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
        """Resolve ``"auto"`` to ``"0"`` (CUDA) or ``"cpu"``.

        Phase 3 will add XPU resolution here.
        """
        if device == "auto":
            return "0" if torch.cuda.is_available() else "cpu"
        return device
