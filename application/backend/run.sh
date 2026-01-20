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
DETECTION_MODEL_BASE_URL="https://storage.geti.intel.com/test-data/geti-tune/models/yolox-s-cards-detection"
DETECTION_MODEL_TARGET_DIR="data/projects/9d6af8e8-6017-4ebe-9126-33aae739c5fa/models/977eeb18-eaac-449d-bc80-e340fbe052ad"

SEGMENTATION_VIDEO_URL="https://storage.geti.intel.com/test-data/geti-tune/media/fish-video.mp4"
SEGMENTATION_VIDEO_TARGET="data/media/fish-video.mp4"
SEGMENTATION_MODEL_BASE_URL="https://storage.geti.intel.com/test-data/geti-tune/models/rtmdet-tiny-fish-segmentation"
SEGMENTATION_MODEL_TARGET_DIR="data/projects/a1b2c3d4-e5f6-7890-abcd-ef1234567890/models/c3d4e5f6-a7b8-9012-cdef-123456789012"

# Model file extensions to download
MODEL_EXTENSIONS=("xml" "bin" "onnx" "ckpt")

# Video URLs and targets
declare -A VIDEO_URLS=(
  ["$DETECTION_VIDEO_TARGET"]="$DETECTION_VIDEO_URL"
  ["$SEGMENTATION_VIDEO_TARGET"]="$SEGMENTATION_VIDEO_URL"
)

# Model base URLs and target directories
declare -A MODEL_CONFIGS=(
  ["$DETECTION_MODEL_TARGET_DIR"]="$DETECTION_MODEL_BASE_URL"
  ["$SEGMENTATION_MODEL_TARGET_DIR"]="$SEGMENTATION_MODEL_BASE_URL"
)

if [[ "$DOWNLOAD_FILES" == "true" ]]; then
  echo "Downloading required files if not present..."

  # Download videos
  for target in "${!VIDEO_URLS[@]}"; do
    if [ ! -f "$target" ]; then
      mkdir -p "$(dirname "$target")"
      echo "Downloading video to $target..."
      curl -fL "${VIDEO_URLS[$target]}" -o "$target"
    else
      echo "Video already exists at $target"
    fi
  done

  # Download model files
  for target_dir in "${!MODEL_CONFIGS[@]}"; do
    mkdir -p "$target_dir"
    base_url="${MODEL_CONFIGS[$target_dir]}"
    for ext in "${MODEL_EXTENSIONS[@]}"; do
      MODEL_FILE="$target_dir/model.$ext"
      if [ ! -f "$MODEL_FILE" ]; then
        echo "Downloading model .$ext file to $target_dir..."
        curl -fL "$base_url.$ext" -o "$MODEL_FILE"
      else
        echo "Model .$ext file already exists at $MODEL_FILE"
      fi
    done
  done
fi

echo "Starting FastAPI server..."

exec $UV_CMD "$APP_MODULE"
