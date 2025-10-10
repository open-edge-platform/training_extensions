#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------------
# run.sh - Script to run the Geti Tune FastAPI server
#
# Features:
# - Optionally seed the database (SEED_DB=true)
# - Optionally download test files (DOWNLOAD_FILES=true)
# - Configure database file name (DATABASE_FILE)
#
# Usage:
#   SEED_DB=true DOWNLOAD_FILES=true ./run.sh                  # Seed and download
#   DATABASE_FILE=geti_tune_e2e.db SEED_DB=true ./run.sh       # Use E2E database
#   ./run.sh                                                   # Run with defaults
#
# Environment variables:
#   SEED_DB           If set to "true", runs database initialization and seeding
#   DOWNLOAD_FILES    If set to "true", downloads test video and model files
#   DATABASE_FILE     Name of the database file (default: geti_tune.db)
#   E2E_ASSETS_S3_URL Base URL for E2E assets
#
# Requirements:
# - 'uv' CLI tool installed and available in PATH
# -----------------------------------------------------------------------------

SEED_DB=${SEED_DB:-false}
DOWNLOAD_FILES=${DOWNLOAD_FILES:-false}
DATABASE_FILE=${DATABASE_FILE:-geti_tune.db}
APP_MODULE=${APP_MODULE:-app/main.py}
UV_CMD=${UV_CMD:-uv run}
E2E_ASSETS_BASE_URL=${E2E_ASSETS_S3_URL:-https://storage.geti.intel.com/test-data/geti-tune}

export PYTHONUNBUFFERED=1
export PYTHONPATH=.
export DATABASE_FILE

DB_PATH="data/${DATABASE_FILE}"

echo "====================================="
echo "Starting Geti Tune Backend"
echo "Database: $DB_PATH"
echo "Assets URL: $E2E_ASSETS_BASE_URL"
echo "====================================="

if [[ "$SEED_DB" == "true" ]]; then
  echo "Seeding the database..."
  # Remove existing database if it exists
  if [ -f "$DB_PATH" ]; then
    echo "Removing existing database: $DB_PATH"
    rm "$DB_PATH"
  fi
  echo "Initializing database..."
  $UV_CMD app/cli.py init-db
  echo "Seeding database with test data..."
  $UV_CMD app/cli.py seed --with-model=True
fi

# URLs and target paths
VIDEO_URL="${E2E_ASSETS_BASE_URL}/media/card-video.mp4"
VIDEO_TARGET="data/media/video.mp4"
MODEL_XML_URL="${E2E_ASSETS_BASE_URL}/models/ssd-card-detection.xml"
MODEL_BIN_URL="${E2E_ASSETS_BASE_URL}/models/ssd-card-detection.bin"
MODEL_TARGET_DIR="data/projects/9d6af8e8-6017-4ebe-9126-33aae739c5fa/models/977eeb18-eaac-449d-bc80-e340fbe052ad"
MODEL_XML_TARGET="$MODEL_TARGET_DIR/model.xml"
MODEL_BIN_TARGET="$MODEL_TARGET_DIR/model.bin"

if [[ "$DOWNLOAD_FILES" == "true" ]]; then
  echo "Downloading required files if not present..."
  # Download video
  if [ ! -f "$VIDEO_TARGET" ]; then
    mkdir -p "$(dirname "$VIDEO_TARGET")"
    echo "Downloading test video from $VIDEO_URL..."
    if curl -fL "$VIDEO_URL" -o "$VIDEO_TARGET"; then
      echo "✓ Video downloaded successfully"
    else
      echo "✗ Failed to download video from $VIDEO_URL"
      exit 1
    fi
  else
    echo "✓ Test video already exists at $VIDEO_TARGET"
  fi
  
  # Verify video file is valid (has content)
  if [ ! -s "$VIDEO_TARGET" ]; then
    echo "✗ Error: Video file is empty at $VIDEO_TARGET"
    exit 1
  fi
  
  # Download model files
  if [ ! -f "$MODEL_XML_TARGET" ]; then
    mkdir -p "$MODEL_TARGET_DIR"
    echo "Downloading model files..."
    if curl -fL "$MODEL_XML_URL" -o "$MODEL_XML_TARGET" && \
       curl -fL "$MODEL_BIN_URL" -o "$MODEL_BIN_TARGET"; then
      echo "✓ Model files downloaded successfully"
    else
      echo "✗ Failed to download model files"
      exit 1
    fi
  else
    echo "✓ Model files already exist"
  fi
fi

echo "Starting FastAPI server..."

exec $UV_CMD "$APP_MODULE"
