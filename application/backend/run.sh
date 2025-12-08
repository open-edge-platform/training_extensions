#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------------
# run.sh - Script to run the Geti Tune FastAPI server
#
# Features:
# - Optionally seed the database before starting the server by setting:
#     SEED_DB=true
# - Optionally download test video and model files before starting the server by setting:
#     DOWNLOAD_FILES=true
#
# Usage:
#   SEED_DB=true DOWNLOAD_FILES=true ./run.sh     # Seed database, download data and launch the server
#   ./run.sh                                      # Run server without seeding or downloading files
#
# Environment variables:
#   SEED_DB         If set to "true", runs `uv run app/cli seed` before starting the server.
#   DOWNLOAD_FILES  If set to "true", downloads test video and model files if not already present.
#   APP_MODULE      Python module to run (default: app/main.py)
#   UV_CMD          Command to launch Uvicorn (default: "uv run")
#
# Requirements:
# - 'uv' CLI tool (Uvicorn) installed and available in PATH
# - Python modules and dependencies installed correctly
# -----------------------------------------------------------------------------

SEED_DB=${SEED_DB:-false}
DOWNLOAD_FILES=${DOWNLOAD_FILES:-false}
APP_MODULE=${APP_MODULE:-app/main.py}
UV_CMD=${UV_CMD:-uv run}

export PYTHONUNBUFFERED=1
export PYTHONPATH=.

if [[ "$SEED_DB" == "true" ]]; then
  echo "Seeding the database..."
  rm data/geti_tune.db || true
  $UV_CMD app/cli.py init-db
  $UV_CMD app/cli.py seed --with-model=True
fi

# URLs and target paths
DETECTION_VIDEO_URL="https://storage.geti.intel.com/test-data/geti-tune/media/card-video.mp4"
DETECTION_VIDEO_TARGET="data/media/card-video.mp4"
DETECTION_MODEL_XML_URL="https://storage.geti.intel.com/test-data/geti-tune/models/ssd-card-detection.xml"
DETECTION_MODEL_BIN_URL="https://storage.geti.intel.com/test-data/geti-tune/models/ssd-card-detection.bin"
DETECTION_MODEL_TARGET_DIR="data/projects/9d6af8e8-6017-4ebe-9126-33aae739c5fa/models/977eeb18-eaac-449d-bc80-e340fbe052ad"
DETECTION_MODEL_XML_TARGET="$DETECTION_MODEL_TARGET_DIR/model.xml"
DETECTION_MODEL_BIN_TARGET="$DETECTION_MODEL_TARGET_DIR/model.bin"

SEGMENTATION_VIDEO_URL="https://storage.geti.intel.com/test-data/geti-tune/media/fish-video.mp4"
SEGMENTATION_VIDEO_TARGET="data/media/fish-video.mp4"
SEGMENTATION_MODEL_XML_URL="https://storage.geti.intel.com/test-data/geti-tune/models/rtmdet-tiny-fish-segmentation.xml"
SEGMENTATION_MODEL_BIN_URL="https://storage.geti.intel.com/test-data/geti-tune/models/rtmdet-tiny-fish-segmentation.bin"
SEGMENTATION_MODEL_TARGET_DIR="data/projects/a1b2c3d4-e5f6-7890-abcd-ef1234567890/models/c3d4e5f6-a7b8-9012-cdef-123456789012"
SEGMENTATION_MODEL_XML_TARGET="$SEGMENTATION_MODEL_TARGET_DIR/model.xml"
SEGMENTATION_MODEL_BIN_TARGET="$SEGMENTATION_MODEL_TARGET_DIR/model.bin"

if [[ "$DOWNLOAD_FILES" == "true" ]]; then
  echo "Downloading required files if not present..."
  # Download detection video
  if [ ! -f "$DETECTION_VIDEO_TARGET" ]; then
    mkdir -p "$(dirname "$DETECTION_VIDEO_TARGET")"
    echo "Downloading test video..."
    curl -fL "$DETECTION_VIDEO_URL" -o "$DETECTION_VIDEO_TARGET"
  else
    echo "Test video already exists at $DETECTION_VIDEO_TARGET"
  fi
  # Download segmentation video
  if [ ! -f "$SEGMENTATION_VIDEO_TARGET" ]; then
    mkdir -p "$(dirname "$SEGMENTATION_VIDEO_TARGET")"
    echo "Downloading test video..."
    curl -fL "$SEGMENTATION_VIDEO_URL" -o "$SEGMENTATION_VIDEO_TARGET"
  else
    echo "Test video already exists at $SEGMENTATION_VIDEO_TARGET"
  fi
  # Download detection model XML
  if [ ! -f "$DETECTION_MODEL_XML_TARGET" ]; then
    mkdir -p "$DETECTION_MODEL_TARGET_DIR"
    echo "Downloading model XML..."
    curl -fL "$DETECTION_MODEL_XML_URL" -o "$DETECTION_MODEL_XML_TARGET"
  else
    echo "Model XML already exists at $DETECTION_MODEL_XML_TARGET"
  fi
  # Download segmentation model XML
  if [ ! -f "$SEGMENTATION_MODEL_XML_TARGET" ]; then
    mkdir -p "$SEGMENTATION_MODEL_TARGET_DIR"
    echo "Downloading model XML..."
    curl -fL "$SEGMENTATION_MODEL_XML_URL" -o "$SEGMENTATION_MODEL_XML_TARGET"
  else
    echo "Model XML already exists at $SEGMENTATION_MODEL_XML_TARGET"
  fi
  # Download detection model BIN
  if [ ! -f "$DETECTION_MODEL_BIN_TARGET" ]; then
    mkdir -p "$DETECTION_MODEL_TARGET_DIR"
    echo "Downloading model BIN..."
    curl -fL "$DETECTION_MODEL_BIN_URL" -o "$DETECTION_MODEL_BIN_TARGET"
  else
    echo "Model BIN already exists at $DETECTION_MODEL_BIN_TARGET"
  fi
  # Download segmentation model BIN
  if [ ! -f "$SEGMENTATION_MODEL_BIN_TARGET" ]; then
    mkdir -p "$SEGMENTATION_MODEL_TARGET_DIR"
    echo "Downloading model BIN..."
    curl -fL "$SEGMENTATION_MODEL_BIN_URL" -o "$SEGMENTATION_MODEL_BIN_TARGET"
  else
    echo "Model BIN already exists at $SEGMENTATION_MODEL_BIN_TARGET"
  fi
fi

echo "Starting FastAPI server..."

exec $UV_CMD "$APP_MODULE"
