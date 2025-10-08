#!/bin/bash
# E2E test backend startup script
# This ensures test assets are in place before seeding

set -euo pipefail

exec 1>&2

echo "============================================="
echo "=== E2E BACKEND SETUP STARTING ==="
echo "============================================="
echo "PWD: $(pwd)"
echo "SEED_DB: ${SEED_DB:-not set}"
echo "E2E_ASSETS_S3_URL: ${E2E_ASSETS_S3_URL:-not set}"
echo "============================================="

echo "=== Preparing test assets ===" 

# Backend data directory
DATA_DIR="$(pwd)/data"
MEDIA_DIR="$DATA_DIR/media"
MODELS_DIR="$DATA_DIR/models"

# Create directories
mkdir -p "$MEDIA_DIR" "$MODELS_DIR"

# Check if we should use S3 or local files
if [ -n "${E2E_ASSETS_S3_URL:-}" ]; then
    echo "Downloading test assets from S3: $E2E_ASSETS_S3_URL"
    
    # Download video
    if [ ! -f "$MEDIA_DIR/video.mp4" ]; then
        echo "Downloading video..."
        curl -L --fail --show-error --silent \
            -o "$MEDIA_DIR/video.mp4" \
            "$E2E_ASSETS_S3_URL/media/sample-video-small.mp4"
        echo "✓ Downloaded video.mp4"
    fi
    
    # Download model files
    for model_file in "ssd-card-detection.bin" "ssd-card-detection.xml"; do
        if [ ! -f "$MODELS_DIR/$model_file" ]; then
            echo "Downloading $model_file..."
            curl -L --fail --show-error --silent \
                -o "$MODELS_DIR/$model_file" \
                "$E2E_ASSETS_S3_URL/models/$model_file"
            echo "✓ Downloaded $model_file"
        fi
    done
    
    echo "✓ All test assets downloaded from S3"
else
    echo "⚠️  E2E_ASSETS_S3_URL not set, assuming assets exist locally"
fi

echo "============================================="
echo "=== E2E Backend Setup: Complete ==="
echo "=== Starting backend with SEED_DB=${SEED_DB:-false} ==="
echo "============================================="

# Now start the backend with seeding
exec ./run.sh
