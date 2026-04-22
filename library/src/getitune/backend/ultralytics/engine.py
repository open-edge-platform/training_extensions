# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Ultralytics engine for getitune."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from getitune.engine.engine import Engine

if TYPE_CHECKING:
    from getitune.data.module import DataModule
    from getitune.types import PathLike
    from getitune.types.types import ANNOTATIONS, DATA, METRICS, MODEL

    from .models.base import UltralyticsModel

logger = logging.getLogger(__name__)


class UltralyticsEngine(Engine):
    """Engine that delegates training, validation, prediction and export to Ultralytics.

    This engine wraps an :class:`UltralyticsModel` (which itself wraps
    ``ultralytics.YOLO``) and a getitune :class:`~getitune.data.module.DataModule`.

    Key design decisions:

    * Images flow through getitune's CPU augmentation pipeline and arrive as
      ``float32 CHW [0, 1]`` tensors.  The data bridge (Phase 2) converts
      these to Ultralytics batch format.  A custom trainer overrides
      ``preprocess_batch()`` to skip the ``/255`` division.
    * Export delegates to Ultralytics' built-in ``model.export(format="openvino")``.
      No ModelAPI metadata embedding in v1 (deferred to Phase 7).
    * CLI integration is deferred — this engine is API-only for v1.
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
        """Initialize the Ultralytics engine.

        Args:
            model: An :class:`UltralyticsModel` instance (detection or instance-seg).
            data: A getitune :class:`~getitune.data.module.DataModule` or path to data root.
            work_dir: Working directory for outputs (checkpoints, exports, logs).
            device: Device string for Ultralytics (e.g. ``"0"`` for GPU 0,
                ``"cpu"``, or ``"auto"``).
            **kwargs: Additional keyword arguments stored as engine overrides.
        """
        from getitune.data.module import DataModule

        from .models.base import UltralyticsModel

        if not isinstance(model, UltralyticsModel):
            msg = f"model must be an UltralyticsModel instance, got {type(model)}"
            raise TypeError(msg)

        self._model = model
        self._work_dir = Path(work_dir)
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._device = self._resolve_device(device)
        self._kwargs = kwargs

        # Data can be a DataModule or a path. If path, store for later.
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
    # Engine ABC implementation
    # ------------------------------------------------------------------

    def train(self, **kwargs) -> METRICS:
        """Train the Ultralytics model.

        Delegates to ``model.yolo.train()`` with the appropriate data
        configuration.  In later phases, this will use a custom trainer
        that bridges getitune's DataModule to Ultralytics format.

        Args:
            **kwargs: Additional overrides forwarded to ``model.yolo.train()``.

        Returns:
            Dictionary of metric name → float value.
        """
        yolo = self._model.yolo
        merged_overrides = self._build_overrides(**kwargs)

        logger.info(
            "Starting Ultralytics training: model=%s, device=%s, imgsz=%d",
            self._model.model_name,
            self._device,
            self._model.imgsz,
        )

        results = yolo.train(
            device=self._device,
            imgsz=self._model.imgsz,
            project=str(self._work_dir),
            name="train",
            exist_ok=True,
            **merged_overrides,
        )

        return self._translate_metrics(results)

    def test(self, **kwargs) -> METRICS:
        """Validate / test the Ultralytics model.

        Args:
            **kwargs: Additional overrides forwarded to ``model.yolo.val()``.

        Returns:
            Dictionary of metric name → float value.
        """
        yolo = self._model.yolo
        merged_overrides = self._build_overrides(**kwargs)

        logger.info("Starting Ultralytics validation: model=%s", self._model.model_name)

        results = yolo.val(
            device=self._device,
            imgsz=self._model.imgsz,
            project=str(self._work_dir),
            name="val",
            exist_ok=True,
            **merged_overrides,
        )

        return self._translate_metrics(results)

    def predict(self, **kwargs) -> ANNOTATIONS:
        """Run inference on the dataset.

        Iterates over the data and converts Ultralytics ``Results`` objects
        into a list of getitune :class:`~getitune.data.entity.sample.Prediction`
        instances.

        Args:
            **kwargs: Additional overrides forwarded to ``model.yolo.predict()``.

        Returns:
            List of :class:`~getitune.data.entity.sample.Prediction` objects,
            one per input image.
        """
        import torch
        from torchvision import tv_tensors

        from getitune.data.entity.base import ImageInfo
        from getitune.data.entity.sample import Prediction

        yolo = self._model.yolo
        merged_overrides = self._build_overrides(**kwargs)

        # Determine prediction source — for now use Ultralytics-native source
        source: str | None = str(merged_overrides.pop("source")) if "source" in merged_overrides else None
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

        predictions: list[Prediction] = []
        for idx, result in enumerate(raw_results):
            # Convert image back to tensor
            img_tensor = torch.from_numpy(result.orig_img).permute(2, 0, 1).float() / 255.0
            h, w = result.orig_shape[0], result.orig_shape[1]
            img_info = ImageInfo(  # pyrefly: ignore[no-matching-overload]
                img_idx=idx,
                img_shape=(h, w),
                ori_shape=(h, w),
            )

            # Extract detection results
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

        return predictions  # type: ignore[return-value]

    def export(self, **kwargs) -> Path:
        """Export the model to OpenVINO IR format.

        Delegates to Ultralytics' built-in ``model.export(format="openvino")``.
        The exported IR files are moved to ``work_dir/exported_model/``.

        Args:
            **kwargs: Additional overrides for ``model.yolo.export()`` (e.g.
                ``half=True`` for FP16).

        Returns:
            Path to the directory containing the exported OpenVINO model.
        """
        yolo = self._model.yolo

        export_dir = yolo.export(
            format="openvino",
            imgsz=self._model.imgsz,
            **kwargs,
        )
        export_path = Path(export_dir)

        # Move to work_dir if not already there
        target_dir = self._work_dir / self._EXPORTED_MODEL_BASE_NAME
        if export_path != target_dir:
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(str(export_path), str(target_dir))
            logger.info("Exported model copied to %s", target_dir)

        return target_dir

    @staticmethod
    def is_supported(model: MODEL, data: DATA) -> bool:
        """Check if this engine supports the given model/data combination.

        Returns ``True`` if ``model`` is an :class:`UltralyticsModel` instance
        and ``data`` is a :class:`~getitune.data.module.DataModule` or path-like.

        Args:
            model: The model to check.
            data: The data to check.

        Returns:
            ``True`` if supported, ``False`` otherwise.
        """
        from getitune.data.module import DataModule

        from .models.base import UltralyticsModel

        return bool(isinstance(model, UltralyticsModel) and isinstance(data, (DataModule, str, os.PathLike)))

    @property
    def work_dir(self) -> PathLike:
        """Get the working directory for the engine."""
        return self._work_dir

    @property
    def model(self) -> UltralyticsModel:
        """Return the Ultralytics model wrapper."""
        return self._model  # type: ignore[return-value]

    @property
    def datamodule(self) -> DATA:
        """Return the datamodule (or data root path) associated with this engine."""
        if self._datamodule is not None:
            return self._datamodule  # type: ignore[return-value]
        if self._data_root is not None:
            return self._data_root  # type: ignore[return-value]
        msg = "No data source configured."
        raise RuntimeError(msg)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_overrides(self, **kwargs) -> dict[str, object]:
        """Merge model-level, engine-level, and call-level overrides.

        Priority (highest wins): call kwargs > engine kwargs > model extra_overrides.
        """
        overrides: dict[str, object] = {}
        overrides.update(self._model.extra_overrides)
        overrides.update(self._kwargs)
        overrides.update(kwargs)
        return overrides

    def _translate_metrics(self, results: object) -> dict[str, float]:
        """Translate Ultralytics results to getitune metric dict.

        Uses the model's ``metric_keys`` mapping.  Unmapped keys are
        passed through with an ``ultralytics/`` prefix.

        Args:
            results: Ultralytics training/validation results object.

        Returns:
            Flat dictionary of metric name → float.
        """
        if results is None:
            return {}

        raw_metrics: dict[str, float] = {}

        # Ultralytics results may be a Results or DetMetrics object
        if hasattr(results, "results_dict"):
            raw_metrics = dict(results.results_dict)
        elif isinstance(results, dict):
            raw_metrics = dict(results)

        translated: dict[str, float] = {}
        for ultra_key, value in raw_metrics.items():
            getitune_key = self._model.metric_keys.get(ultra_key, f"ultralytics/{ultra_key}")
            translated[getitune_key] = float(value) if not isinstance(value, float) else value

        return translated

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve ``"auto"`` device to a concrete Ultralytics device string.

        Args:
            device: Device string (``"auto"``, ``"cpu"``, ``"0"``, ``"0,1"``, etc.)

        Returns:
            Resolved device string.
        """
        if device == "auto":
            import torch

            return "0" if torch.cuda.is_available() else "cpu"
        return device
