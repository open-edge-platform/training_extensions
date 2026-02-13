#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------------
# run.sh - Script to run the Geti Tune FastAPI server
#
# Features:
# - Optionally set up demo projects before starting the server by using:
#     --setup-demo
#   This will delete all existing data, reset the database to a clean state,
#   and import demo projects from S3.
#
# Usage:
#   ./run.sh --setup-demo                    # Reset database and set up demo projects, then launch the server
#   ./run.sh --setup-demo --force-import    # Same as above, but bypass schema version checks during import
#   ./run.sh                                # Run server with existing data
#
# Arguments:
#   --setup-demo    Deletes all existing data, resets the database,
#                   and imports demo projects before starting the server.
#   --force-import  Bypasses database schema version checks during import.
#                   Use with caution as this may cause import errors or data corruption.
#                   (Only valid when used with --setup-demo)
#
# Environment variables:
#   APP_MODULE      Python module to run (default: app/main.py)
#   UV_CMD          Command to launch Uvicorn (default: "uv run")
#
# Requirements:
# - 'uv' CLI tool (Uvicorn) installed and available in PATH
# - Python modules and dependencies installed correctly
# -----------------------------------------------------------------------------

# Default values
SETUP_DEMO=false
FORCE_IMPORT=false
APP_MODULE=${APP_MODULE:-app/main.py}
UV_CMD=${UV_CMD:-uv run}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --setup-demo)
      SETUP_DEMO=true
      shift
      ;;
    --force-import)
      FORCE_IMPORT=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--setup-demo] [--force-import]"
      echo ""
      echo "Options:"
      echo "  --setup-demo    Reset database and import demo projects"
      echo "  --force-import  Bypass schema version checks (use with --setup-demo)"
      echo "  -h, --help      Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Validate arguments
if [[ "$FORCE_IMPORT" == "true" && "$SETUP_DEMO" != "true" ]]; then
  echo "Error: --force-import can only be used with --setup-demo"
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH=.

# Demo project archives hosted in S3
DEMO_PROJECT_URLS=(
  "https://storage.geti.intel.com/test-data/geti/demo-projects/pre-release/apples-tomatoes-classification.zip"
  "https://storage.geti.intel.com/test-data/geti/demo-projects/pre-release/horses-segmentation.zip"
  "https://storage.geti.intel.com/test-data/geti/demo-projects/pre-release/airplanes-segmentation.zip"
)

# Demo video sources hosted in S3
DEMO_VIDEO_URLS=(
  "https://storage.geti.intel.com/test-data/geti/demo-videos/pre-release/apples.mp4"
  "https://storage.geti.intel.com/test-data/geti/demo-videos/pre-release/horses.mp4"
  "https://storage.geti.intel.com/test-data/geti/demo-videos/pre-release/airplanes.mp4"
)

# Directory for downloaded demo videos
DEMO_VIDEOS_DIR="data/.demo_videos"

# Temporary directory for downloaded archives
DEMO_ARCHIVES_DIR="data/.demo_archives"

# Function to download and import a demo project
import_demo_project() {
  local url="$1"
  local filename
  filename=$(basename "$url")
  local archive_path="$DEMO_ARCHIVES_DIR/$filename"

  # Download archive if not already present
  if [ ! -f "$archive_path" ]; then
    echo "Downloading demo project archive: $filename..."
    curl -fL "$url" -o "$archive_path"
  else
    echo "Demo project archive already downloaded: $filename"
  fi

  # Import the project with optional force flag
  echo "Importing demo project from: $filename..."
  local import_cmd="$UV_CMD app/cli.py import-project --input $archive_path"
  if [[ "$FORCE_IMPORT" == "true" ]]; then
    import_cmd="$import_cmd --force-import"
    echo "  (using --force-import flag to bypass schema version checks)"
  fi

  $import_cmd
}

# Function to download a demo video
download_demo_video() {
  local url="$1"
  local filename
  filename=$(basename "$url")
  local video_path="$DEMO_VIDEOS_DIR/$filename"

  # Download video if not already present
  if [ ! -f "$video_path" ]; then
    echo "Downloading demo video: $filename..." >&2
    if ! curl -fL "$url" -o "$video_path"; then
      echo "Error: Failed to download demo video from: $url" >&2
      exit 1
    fi
  else
    echo "Demo video already downloaded: $filename" >&2
  fi

  # Return the local path (only this goes to stdout)
  echo "$video_path"
}

# Function to setup demo sources
setup_demo_sources() {
  echo "Setting up demo video sources..."

  # Create directory for demo videos
  mkdir -p "$DEMO_VIDEOS_DIR"

  # Download all videos and collect local paths
  local video_paths=()
  for url in "${DEMO_VIDEO_URLS[@]}"; do
    local video_path
    video_path=$(download_demo_video "$url")
    video_paths+=("$video_path")
  done

  # Build the CLI command with local file paths
  local cmd="$UV_CMD app/cli.py setup-demo-sources"
  for path in "${video_paths[@]}"; do
    cmd="$cmd --video-path $path"
  done

  $cmd
}

if [[ "$SETUP_DEMO" == "true" ]]; then
  echo "Setting up demo projects..."
  echo "WARNING: This will delete all existing data and reset the database!"
  read -p "Do you want to continue? (yes/no): " -r
  echo
  if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
    echo "Demo setup cancelled. Starting server with existing data..."
  else
    # Remove existing database and projects to start fresh
    rm -f data/geti_tune.db
    rm -rf data/projects/*
    rm -rf data/output/*
    rm -rf data/demo_videos/*

    # Initialize the database
    $UV_CMD app/cli.py init-db

    # Create directory for downloaded archives
    mkdir -p "$DEMO_ARCHIVES_DIR"

    # Download and import each demo project
    for url in "${DEMO_PROJECT_URLS[@]}"; do
      import_demo_project "$url"
    done

    # Setup demo sources (videos)
    setup_demo_sources

    echo "Demo setup complete."
  fi
fi

echo "Starting FastAPI server..."

exec $UV_CMD "$APP_MODULE"
