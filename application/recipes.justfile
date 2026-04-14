set unstable

# Demo project archives hosted in S3
demo_project_urls := "https://storage.geti.intel.com/test-data/geti/demo-projects/pre-release/flowers-classification.zip https://storage.geti.intel.com/test-data/geti/demo-projects/pre-release/airplanes-detection.zip https://storage.geti.intel.com/test-data/geti/demo-projects/pre-release/horses-segmentation.zip"

# Demo video sources hosted in S3
demo_video_urls := "https://storage.geti.intel.com/test-data/geti/demo-videos/pre-release/sunflower.mp4 https://storage.geti.intel.com/test-data/geti/demo-videos/pre-release/horses.mp4 https://storage.geti.intel.com/test-data/geti/demo-videos/pre-release/airplanes.mp4"

# Cache directories
demo_archives_dir := "data/.demo_cache/archives"
demo_videos_dir := "data/.demo_cache/videos"

# Install uv
install-uv:
    #!/usr/bin/env bash
    set -euo pipefail
    REPO_ROOT=$(git rev-parse --show-toplevel)
    # Extract the version from required-version, stripping any PEP 440 specifier prefix and CR/LF
    UV_VERSION=$(grep -A 3 '\[tool\.uv\]' "${REPO_ROOT}/application/backend/pyproject.toml" | grep 'required-version' | sed -E 's/.*=[[:space:]]*"[^0-9]*([0-9]+\.[0-9]+\.[0-9]+).*/\1/' | tr -d '\r')
    if [ -z "$UV_VERSION" ]; then
        echo "Error: could not parse uv version from pyproject.toml" >&2
        exit 1
    fi
    if command -v uv > /dev/null; then
        INSTALLED_VERSION=$(uv --version | awk '{print $2}' | tr -d '\r')
        if [ "$INSTALLED_VERSION" = "$UV_VERSION" ]; then
            exit 0
        else
            echo "uv version mismatch: installed=${INSTALLED_VERSION}, required~=${UV_VERSION}. Reinstalling..."
        fi
    fi
    curl --proto '=https' --tlsv1.2 -LsSf "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-installer.sh" | sh

# Download a file from a URL to a destination directory (skips if already present)
[private]
download url dir:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "{{ dir }}"
    filename=$(basename "{{ url }}")
    dest="{{ dir }}/$filename"
    if [ ! -f "$dest" ]; then
        echo "Downloading: $filename..."
        curl -fL "{{ url }}" -o "$dest"
    else
        echo "Already downloaded: $filename"
    fi

# Import a single demo project archive, with optional --force-import flag
[private]
import-project archive force_flag="":
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Importing demo project from: $(basename "{{ archive }}")..."
    cmd="uv run app/cli.py import-project --input {{ archive }}"
    if [ -n "{{ force_flag }}" ]; then
        cmd="$cmd --force-import"
        echo "  (using --force-import flag to bypass schema version checks)"
    fi
    PYTHONPATH=. $cmd

# Download and import all demo projects
[private]
import-demo-projects force_flag="":
    #!/usr/bin/env bash
    set -euo pipefail
    for url in {{ demo_project_urls }}; do
        just download "$url" "{{ demo_archives_dir }}"
        filename=$(basename "$url")
        just import-project "{{ demo_archives_dir }}/$filename" "{{ force_flag }}"
    done

# Download all demo videos and set up demo sources via CLI
[private]
setup-demo-sources:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Setting up demo video sources..."
    video_args=""
    for url in {{ demo_video_urls }}; do
        just download "$url" "{{ demo_videos_dir }}"
        filename=$(basename "$url")
        video_args="$video_args --video-path {{ demo_videos_dir }}/$filename"
    done
    PYTHONPATH=. uv run app/cli.py setup-demo-sources $video_args

# Set up demo sinks via CLI
[private]
setup-demo-sinks:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Setting up demo sinks..."
    PYTHONPATH=. uv run app/cli.py setup-demo-sinks
