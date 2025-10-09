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
VIDEO_URL="https://storage.geti.intel.com/test-data/geti-tune/media/card-video.mp4"
VIDEO_TARGET="data/media/video.mp4"
MODEL_XML_URL="https://storage.geti.intel.com/test-data/geti-tune/models/ssd-card-detection.xml"
MODEL_BIN_URL="https://storage.geti.intel.com/test-data/geti-tune/models/ssd-card-detection.bin"
MODEL_TARGET_DIR="data/projects/9d6af8e8-6017-4ebe-9126-33aae739c5fa/models/977eeb18-eaac-449d-bc80-e340fbe052ad"
MODEL_XML_TARGET="$MODEL_TARGET_DIR/model.xml"
MODEL_BIN_TARGET="$MODEL_TARGET_DIR/model.bin"

if [[ "$DOWNLOAD_FILES" == "true" ]]; then
  echo "Downloading required files if not present..."
  # Download video
  if [ ! -f "$VIDEO_TARGET" ]; then
    mkdir -p "$(dirname "$VIDEO_TARGET")"
    echo "Downloading test video..."
    curl -fL "$VIDEO_URL" -o "$VIDEO_TARGET"
  else
    echo "Test video already exists at $VIDEO_TARGET"
  fi
  # Download model XML
  if [ ! -f "$MODEL_XML_TARGET" ]; then
    mkdir -p "$MODEL_TARGET_DIR"
    echo "Downloading model XML..."
    curl -fL "$MODEL_XML_URL" -o "$MODEL_XML_TARGET"
  else
    echo "Model XML already exists at $MODEL_XML_TARGET"
  fi
  # Download model BIN
  if [ ! -f "$MODEL_BIN_TARGET" ]; then
    mkdir -p "$MODEL_TARGET_DIR"
    echo "Downloading model BIN..."
    curl -fL "$MODEL_BIN_URL" -o "$MODEL_BIN_TARGET"
  else
    echo "Model BIN already exists at $MODEL_BIN_TARGET"
  fi
fi

echo "Starting FastAPI server..."

exec $UV_CMD "$APP_MODULE"
