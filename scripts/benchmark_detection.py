# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Benchmark script: Ultralytics vs getitune detection training.

Three sections:
  1. Raw Ultralytics API (YOLO11s on WGISD in YOLO format)
  2. getitune Lightning backend (YOLOX-S on WGISD in COCO format)
  3. Future getitune + Ultralytics backend (target API — not yet implemented)

Usage:
    # Run all sections (section 3 will be skipped with a message)
    python scripts/benchmark_detection.py --all

    # Run only Ultralytics
    python scripts/benchmark_detection.py --ultralytics

    # Run only getitune Lightning
    python scripts/benchmark_detection.py --getitune

    # Run only the future API demo (will fail until backend is implemented)
    python scripts/benchmark_detection.py --future
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COCO_DATA_ROOT = Path("/home/kprokofi/bench_data/detection/wgisd_merged_coco_small")
YOLO_DATA_ROOT = Path("/home/kprokofi/bench_data/detection/wgisd_yolo")
YOLO_DATASET_YAML = YOLO_DATA_ROOT / "dataset.yaml"
WORK_DIR = Path("/home/kprokofi/bench_data/detection/bench_outputs")

RECIPE_YOLOX_S = (
    Path(__file__).resolve().parent.parent / "library" / "src" / "getitune" / "recipe" / "detection" / "yolox_s.yaml"
)

# Training params (shared where possible)
EPOCHS = 5
BATCH_SIZE = 8
IMG_SIZE = 640
DEVICE = "0"  # GPU device index for Ultralytics
DEVICE_GETITUNE = "gpu"  # Device type for getitune (enum: auto, gpu, cpu, xpu)


# ===================================================================
# Section 1: Raw Ultralytics API
# ===================================================================
def run_ultralytics() -> None:
    """Train, validate, predict, and export using the Ultralytics Python API."""
    from ultralytics import YOLO

    logger.info("=" * 60)
    logger.info("SECTION 1: Raw Ultralytics API (YOLO11s)")
    logger.info("=" * 60)

    work = WORK_DIR / "ultralytics"
    work.mkdir(parents=True, exist_ok=True)

    # --- Train --------------------------------------------------------
    model = YOLO("yolo11s.pt")  # pretrained COCO weights
    logger.info("Starting Ultralytics training...")
    t0 = time.perf_counter()

    model.train(
        data=str(YOLO_DATASET_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(work),
        name="train",
        exist_ok=True,
        verbose=True,
    )
    train_time = time.perf_counter() - t0
    logger.info(f"Ultralytics training done in {train_time:.1f}s")

    # --- Validate -----------------------------------------------------
    logger.info("Running Ultralytics validation...")
    val_metrics = model.val(
        data=str(YOLO_DATASET_YAML),
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(work),
        name="val",
        exist_ok=True,
    )
    logger.info(f"Ultralytics val mAP50: {val_metrics.box.map50:.4f}")
    logger.info(f"Ultralytics val mAP50-95: {val_metrics.box.map:.4f}")

    # --- Predict ------------------------------------------------------
    logger.info("Running Ultralytics prediction on val split...")
    sample_images = list((YOLO_DATA_ROOT / "images" / "val").glob("*.jpg"))[:3]
    results = model.predict(
        source=sample_images,
        imgsz=IMG_SIZE,
        device=DEVICE,
        save=True,
        project=str(work),
        name="predict",
        exist_ok=True,
    )
    for r in results:
        n_boxes = len(r.boxes)
        logger.info(f"  {Path(r.path).name}: {n_boxes} detections")

    # --- Export to OpenVINO -------------------------------------------
    logger.info("Exporting Ultralytics model to OpenVINO IR...")
    export_path = model.export(format="openvino", imgsz=IMG_SIZE, half=False)
    logger.info(f"Ultralytics export done: {export_path}")

    logger.info("Section 1 complete.\n")


# ===================================================================
# Section 2: getitune Lightning Backend (YOLOX-S)
# ===================================================================
def run_getitune_lightning() -> None:
    """Train, test, predict, and export using getitune's LightningEngine."""
    from getitune.backend.lightning.engine import LightningEngine
    from getitune.engine import create_engine
    from getitune.types.export import ExportFormat
    from getitune.types.precision import Precision

    logger.info("=" * 60)
    logger.info("SECTION 2: getitune Lightning Backend (YOLOX-S)")
    logger.info("=" * 60)

    work = WORK_DIR / "getitune_lightning"
    work.mkdir(parents=True, exist_ok=True)

    # --- Build engine from recipe + COCO dataset ----------------------
    logger.info(f"Building LightningEngine from recipe: {RECIPE_YOLOX_S}")
    engine = LightningEngine.from_config(
        config_path=str(RECIPE_YOLOX_S),
        data_root=str(COCO_DATA_ROOT),
        work_dir=str(work),
        device=DEVICE_GETITUNE,
    )
    logger.info(f"Model: {engine.model.__class__.__name__}")
    logger.info(f"DataModule labels: {engine.datamodule.label_info}")

    # --- Train --------------------------------------------------------
    logger.info("Starting getitune Lightning training...")
    t0 = time.perf_counter()
    train_metrics = engine.train(max_epochs=EPOCHS)
    train_time = time.perf_counter() - t0
    logger.info(f"getitune training done in {train_time:.1f}s")
    logger.info(f"Train metrics: {train_metrics}")

    # --- Test (evaluate) ----------------------------------------------
    logger.info("Running getitune test (evaluation)...")
    test_metrics = engine.test()
    logger.info(f"Test metrics: {test_metrics}")

    # --- Predict ------------------------------------------------------
    logger.info("Running getitune prediction...")
    predictions = engine.predict()
    logger.info(f"Predictions: {len(predictions)} samples")
    for pred in predictions[:3]:
        n_boxes = len(pred.bboxes) if hasattr(pred, "bboxes") and pred.bboxes is not None else 0
        logger.info(f"  Sample: {n_boxes} detections")

    # --- Export to OpenVINO -------------------------------------------
    logger.info("Exporting getitune model to OpenVINO IR...")
    ov_path = engine.export(
        export_format=ExportFormat.OPENVINO,
        export_precision=Precision.FP32,
    )
    logger.info(f"getitune OV export done: {ov_path}")

    # --- OV inference -------------------------------------------------
    logger.info("Running OV inference via create_engine...")
    ov_engine = create_engine(
        model=ov_path,
        data=engine.datamodule,
        work_dir=str(work / "ov_inference"),
    )
    ov_metrics = ov_engine.test()
    logger.info(f"OV test metrics: {ov_metrics}")

    logger.info("Section 2 complete.\n")


# ===================================================================
# Section 3: Future getitune + Ultralytics Backend (target API)
# ===================================================================
def run_future_ultralytics_backend() -> None:
    """Target API for the Ultralytics backend in getitune.

    This demonstrates what the API SHOULD look like once the backend is
    implemented. It will fail until Phase 1-2 of the integration plan
    are complete.
    """
    logger.info("=" * 60)
    logger.info("SECTION 3: Future getitune + Ultralytics Backend")
    logger.info("=" * 60)

    work = WORK_DIR / "getitune_ultralytics"
    work.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Option A: Engine from recipe YAML (the primary path)
    #
    # The recipe YAML would look like:
    #   task: DETECTION
    #   model:
    #     class_path: getitune.backend.ultralytics.models.detection.UltralyticsDetectionModel
    #     init_args:
    #       model_name: yolo11s.pt
    #       label_info: 5
    #   data: ../_base_/data/ultralytics_detection.yaml
    #   ...
    #
    # Usage would be identical to Lightning — create_engine picks
    # UltralyticsEngine automatically based on model type.
    # ------------------------------------------------------------------

    try:
        from getitune.engine import create_engine
        from getitune.types.export import ExportFormat
        from getitune.types.precision import Precision

        # This recipe does not exist yet — it will be created in Phase 4
        recipe_path = (
            Path(__file__).resolve().parent.parent
            / "library"
            / "src"
            / "getitune"
            / "recipe"
            / "detection"
            / "yolo11_s.yaml"
        )

        if not recipe_path.exists():
            logger.warning(f"Recipe not found: {recipe_path}")
            logger.info("Falling back to manual engine construction (Option B)...")
            raise FileNotFoundError(recipe_path)

        # Option A: from_config (same API as Lightning)
        from getitune.backend.ultralytics.engine import UltralyticsEngine

        engine = UltralyticsEngine.from_config(
            config_path=str(recipe_path),
            data_root=str(COCO_DATA_ROOT),
            work_dir=str(work),
            device=DEVICE_GETITUNE,
        )

    except (ImportError, FileNotFoundError):
        # ------------------------------------------------------------------
        # Option B: Manual construction (works without recipe YAML)
        #
        # This is the programmatic API — useful for notebooks and scripts.
        # ------------------------------------------------------------------
        logger.info("Attempting manual engine construction...")

        try:
            from getitune.backend.ultralytics.engine import UltralyticsEngine
            from getitune.backend.ultralytics.models.detection import UltralyticsDetectionModel
            from getitune.data.module import DataModule
            from getitune.types.export import ExportFormat
            from getitune.types.precision import Precision
            from getitune.types.task import TaskType
        except ImportError as e:
            logger.error(f"Ultralytics backend not yet implemented: {e}")
            logger.info(
                "This section will work after Phases 1-2 of the integration plan.\nShowing target API as comments only."
            )
            _print_target_api()
            return

        # Build DataModule from COCO dataset (same as Lightning path)
        datamodule = DataModule(
            task=TaskType.DETECTION,
            data_root=str(COCO_DATA_ROOT),
            input_size=(IMG_SIZE, IMG_SIZE),
        )

        # Build Ultralytics model wrapper
        model = UltralyticsDetectionModel(
            model_name="yolo11s.pt",
            label_info=datamodule.label_info,
        )

        # create_engine should auto-select UltralyticsEngine
        engine = create_engine(
            model=model,
            data=datamodule,
            work_dir=str(work),
            device=DEVICE_GETITUNE,
        )
        assert isinstance(engine, UltralyticsEngine), f"Expected UltralyticsEngine, got {type(engine)}"

    # ------------------------------------------------------------------
    # From here on, the API is IDENTICAL to Lightning (Section 2)
    # ------------------------------------------------------------------
    logger.info(f"Engine type: {engine.__class__.__name__}")
    logger.info(f"Model: {engine.model}")

    # Train
    logger.info("Starting training...")
    t0 = time.perf_counter()
    train_metrics = engine.train(max_epochs=EPOCHS)
    train_time = time.perf_counter() - t0
    logger.info(f"Training done in {train_time:.1f}s")
    logger.info(f"Train metrics: {train_metrics}")

    # Test
    logger.info("Running test (evaluation)...")
    test_metrics = engine.test()
    logger.info(f"Test metrics: {test_metrics}")

    # Predict
    logger.info("Running prediction...")
    predictions = engine.predict()
    logger.info(f"Predictions: {len(predictions)} samples")
    for pred in predictions[:3]:
        n_boxes = len(pred.bboxes) if hasattr(pred, "bboxes") and pred.bboxes is not None else 0
        logger.info(f"  Sample: {n_boxes} detections")

    # Export to OpenVINO
    logger.info("Exporting to OpenVINO IR...")
    ov_path = engine.export(
        export_format=ExportFormat.OPENVINO,
        export_precision=Precision.FP32,
    )
    logger.info(f"Export done: {ov_path}")

    # OV inference on exported model
    logger.info("Running OV inference on exported model...")
    ov_engine = create_engine(
        model=ov_path,
        data=engine.datamodule,
        work_dir=str(work / "ov_inference"),
    )
    ov_metrics = ov_engine.test()
    logger.info(f"OV test metrics: {ov_metrics}")

    logger.info("Section 3 complete.\n")


def _print_target_api() -> None:
    """Print the target API as documentation when imports fail."""
    api_doc = """
    # ================================================================
    # TARGET API — getitune + Ultralytics Backend
    # ================================================================
    #
    # Once implemented, usage is identical to the Lightning backend:
    #
    #   from getitune.engine import create_engine
    #   from getitune.backend.ultralytics.engine import UltralyticsEngine
    #   from getitune.backend.ultralytics.models.detection import UltralyticsDetectionModel
    #   from getitune.data.module import DataModule
    #   from getitune.types.task import TaskType
    #   from getitune.types.export import ExportFormat
    #   from getitune.types.precision import Precision
    #
    #   # Option A: From recipe YAML (recommended)
    #   engine = UltralyticsEngine.from_config(
    #       config_path="recipe/detection/yolo11_s.yaml",
    #       data_root="/path/to/coco_dataset",
    #       work_dir="/path/to/output",
    #       device="0",
    #   )
    #
    #   # Option B: Programmatic construction
    #   datamodule = DataModule(
    #       task=TaskType.DETECTION,
    #       data_root="/path/to/coco_dataset",
    #       input_size=(640, 640),
    #   )
    #   model = UltralyticsDetectionModel(
    #       model_name="yolo11s.pt",
    #       label_info=datamodule.label_info,
    #   )
    #   engine = create_engine(model=model, data=datamodule, work_dir="...", device="0")
    #
    #   # Train / test / predict / export — same API as Lightning
    #   train_metrics = engine.train(max_epochs=5)
    #   test_metrics  = engine.test()
    #   predictions   = engine.predict()
    #   ov_path       = engine.export(ExportFormat.OPENVINO, Precision.FP32)
    #
    #   # OV inference on exported model — works via create_engine
    #   ov_engine = create_engine(model=ov_path, data=datamodule, work_dir="...")
    #   ov_metrics = ov_engine.test()
    #
    # Key design points:
    #   - create_engine() auto-selects UltralyticsEngine for UltralyticsModel
    #   - DataModule is shared — same COCO dataset, same augmentations
    #   - train/test/predict/export API is identical to LightningEngine
    #   - Ultralytics augmentations are disabled; getitune CPU pipeline is used
    #   - preprocess_batch() is overridden to skip /255 (data is already [0,1])
    # ================================================================
    """
    logger.info(api_doc)


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    global EPOCHS, DEVICE  # noqa: PLW0603

    parser = argparse.ArgumentParser(description="Detection benchmark: Ultralytics vs getitune")
    parser.add_argument("--ultralytics", action="store_true", help="Run Section 1: Raw Ultralytics API")
    parser.add_argument("--getitune", action="store_true", help="Run Section 2: getitune Lightning (YOLOX-S)")
    parser.add_argument("--future", action="store_true", help="Run Section 3: Future Ultralytics backend")
    parser.add_argument("--all", action="store_true", help="Run all sections")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Number of training epochs (default: {EPOCHS})")
    parser.add_argument("--device", type=str, default=DEVICE, help=f"GPU device (default: {DEVICE})")
    args = parser.parse_args()

    EPOCHS = args.epochs
    DEVICE = args.device

    run_any = args.all or args.ultralytics or args.getitune or args.future
    if not run_any:
        parser.print_help()
        return

    if args.all or args.ultralytics:
        run_ultralytics()

    if args.all or args.getitune:
        run_getitune_lightning()

    if args.all or args.future:
        run_future_ultralytics_backend()

    logger.info("Benchmark complete.")


if __name__ == "__main__":
    main()
