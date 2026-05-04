# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Class-based exporter for Ultralytics YOLO models.

Follows the same architecture as
:class:`~getitune.backend.lightning.exporter.native.LightningModelExporter`
— inherits from :class:`~getitune.backend.lightning.exporter.base.ModelExporter`
and reuses its metadata-embedding, preprocessing-parameter and postprocessing
helpers.

Key differences from the Lightning path:
  * The raw export is performed by ``ultralytics.YOLO.export()`` instead of
    ``torch.onnx.export`` / ``openvino.convert_model``.
  * FP16 is handled by ``openvino.save_model(compress_to_fp16=True)`` (OpenVINO)
    or ``onnxconverter_common`` (ONNX) — **no** manual Cast-node insertion.
  * The Ultralytics export always runs in **FP32**; weight compression happens
    as a post-step, matching the Lightning contract exactly.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import onnx
import openvino

from getitune.backend.lightning.exporter.base import ModelExporter
from getitune.types.export import TaskLevelExportParameters
from getitune.types.precision import Precision

if TYPE_CHECKING:
    from ultralytics import YOLO

    from getitune.backend.lightning.models.base import DataInputParams

logger = logging.getLogger(__name__)


class UltralyticsModelExporter(ModelExporter):
    """Exporter for Ultralytics YOLO models.

    Delegates the actual model conversion to ``ultralytics.YOLO.export()``,
    then applies the same metadata-embedding and FP16-compression pipeline
    as :class:`LightningModelExporter`.

    Args:
        task_level_export_parameters: Collection of export parameters
            (model type, task type, labels, thresholds, etc.).
        data_input_params: Data input parameters for preprocessing metadata
            (mean, std, input_size).
        resize_mode: Resize strategy embedded in export metadata.
        pad_value: Padding value for letterbox resize.
        swap_rgb: Whether the model expects RGB input (True for YOLO).
        output_names: Optional output tensor names to embed.
        input_names: Optional input tensor names to embed.
    """

    def __init__(
        self,
        task_level_export_parameters: TaskLevelExportParameters,
        data_input_params: DataInputParams,
        resize_mode: Literal[
            "crop", "standard", "fit_to_window", "fit_to_window_letterbox"
        ] = "fit_to_window_letterbox",
        pad_value: int = 114,
        swap_rgb: bool = True,
        output_names: list[str] | None = None,
        input_names: list[str] | None = None,
    ) -> None:
        super().__init__(
            task_level_export_parameters=task_level_export_parameters,
            data_input_params=data_input_params,
            resize_mode=resize_mode,
            pad_value=pad_value,
            swap_rgb=swap_rgb,
            output_names=output_names,
            input_names=input_names,
        )

    # ------------------------------------------------------------------
    # Public API (overrides from ModelExporter)
    # ------------------------------------------------------------------

    def to_openvino(
        self,
        model: YOLO,  # type: ignore[override]
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: Precision = Precision.FP32,
    ) -> Path:
        """Export a YOLO model to OpenVINO IR format.

        1. Export FP32 via ``model.export(format="openvino", half=False, end2end=False)``.
        2. Load the resulting OV model and apply inherited metadata embedding
           (preprocessing params from ``DataInputParams`` + ``TaskLevelExportParameters``).
        3. Save with ``compress_to_fp16=True`` when *precision* is FP16 —
           **no manual Cast nodes**.
        4. Clean up raw Ultralytics export artefacts.

        Args:
            model: Ultralytics ``YOLO`` instance to export.
            output_dir: Directory for the final ``<base_model_name>.xml`` file.
            base_model_name: Stem of the output file.
            precision: ``FP32`` or ``FP16``.

        Returns:
            Path to the exported ``.xml`` file.
        """
        imgsz = self.data_input_params.input_size[0]

        # Step 1 — Ultralytics raw FP32 export
        raw_result = model.export(
            format="openvino",
            imgsz=imgsz,
            half=False,
            end2end=False,
        )
        raw_path = Path(raw_result)

        # Step 2 — Find the .xml inside the raw export directory
        raw_xml = self._find_xml_in_export(raw_path)

        # Step 3 — Load, postprocess (metadata + names), save
        ov_model = openvino.Core().read_model(str(raw_xml))
        ov_model = self._postprocess_openvino_model(ov_model)

        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / (base_model_name + ".xml")
        openvino.save_model(ov_model, str(save_path), compress_to_fp16=(precision == Precision.FP16))

        logger.info(
            "Ultralytics OpenVINO export done (%d inputs, %d outputs) -> %s",
            len(ov_model.inputs),
            len(ov_model.outputs),
            save_path,
        )

        # Step 4 — Clean up Ultralytics raw export artefacts
        self._cleanup_raw_export(raw_path, save_path.parent)

        return save_path

    def to_onnx(
        self,
        model: YOLO,  # type: ignore[override]
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: Precision = Precision.FP32,
        embed_metadata: bool = True,
    ) -> Path:
        """Export a YOLO model to ONNX format.

        1. Export FP32 via ``model.export(format="onnx", half=False, end2end=False)``.
        2. Load ONNX, apply inherited metadata embedding + FP16 conversion
           (via ``onnxconverter_common``, same as Lightning).
        3. Save to target location.

        Args:
            model: Ultralytics ``YOLO`` instance to export.
            output_dir: Directory for the final ``.onnx`` file.
            base_model_name: Stem of the output file.
            precision: ``FP32`` or ``FP16``.
            embed_metadata: Whether to embed metadata into ONNX model.

        Returns:
            Path to the exported ``.onnx`` file.
        """
        imgsz = self.data_input_params.input_size[0]

        # Step 1 — Ultralytics raw FP32 export
        raw_result = model.export(
            format="onnx",
            imgsz=imgsz,
            half=False,
            end2end=False,
        )
        raw_path = Path(raw_result)

        # Step 2 — Load, postprocess (metadata + FP16), save
        onnx_model = onnx.load(str(raw_path))
        onnx_model = self._postprocess_onnx_model(onnx_model, embed_metadata, precision)

        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / (base_model_name + ".onnx")
        onnx.save(onnx_model, str(save_path))

        logger.info("Ultralytics ONNX export done -> %s", save_path)

        # Step 3 — Clean up raw export if different from target
        if raw_path.resolve() != save_path.resolve():
            raw_path.unlink(missing_ok=True)

        return save_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_xml_in_export(export_path: Path) -> Path:
        """Locate the ``.xml`` file inside an Ultralytics OpenVINO export.

        Ultralytics typically writes to a directory (e.g. ``model_openvino/``).
        This helper finds the first ``.xml`` file inside that directory, or
        returns *export_path* itself if it is already an XML file.
        """
        if export_path.is_file() and export_path.suffix == ".xml":
            return export_path

        if export_path.is_dir():
            xml_files = list(export_path.glob("*.xml"))
            if xml_files:
                return xml_files[0]

        msg = f"No .xml file found in Ultralytics export output: {export_path}"
        raise FileNotFoundError(msg)

    @staticmethod
    def _cleanup_raw_export(raw_path: Path, target_dir: Path) -> None:
        """Remove Ultralytics raw export artefacts if they differ from target.

        Ultralytics creates a directory like ``runs/detect/export/model_openvino/``.
        After we have copied the final model to *target_dir*, remove the raw
        directory to avoid stale artefacts.
        """
        if not raw_path.exists():
            return

        raw_resolved = raw_path.resolve()
        target_resolved = target_dir.resolve()

        # Don't delete if the raw export IS the target
        if raw_resolved == target_resolved:
            return
        # Don't delete if the raw export is a parent of the target
        if target_resolved.is_relative_to(raw_resolved):
            return

        if raw_path.is_dir():
            shutil.rmtree(raw_path, ignore_errors=True)
        elif raw_path.is_file():
            raw_path.unlink(missing_ok=True)
