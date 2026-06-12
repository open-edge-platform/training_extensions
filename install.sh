#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "ERROR: Installation failed at line $1 (exit code $exit_code)."
        if [ -n "${LOG_FILE:-}" ] && [ -f "$LOG_FILE" ]; then
            echo "Check $LOG_FILE for details."
        fi
        echo "Re-run with --verbose for more details."
    fi
    exit $exit_code
}

trap 'cleanup $LINENO' ERR
trap 'echo ""; echo "Installation interrupted."; exit 130' INT TERM

GIT_URL="https://github.com/open-edge-platform/training_extensions.git"
GIT_BRANCH="app/v3.0.0rc3"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Install Intel Geti application and its dependencies.

Options:
  -v, --verbose     Show detailed output from all commands
  -y, --yes         Assume yes to all prompts (non-interactive mode)
  -w, --work-dir    Set the working directory (default: \$HOME/geti)
  -h, --help        Show this help message and exit
EOF
}

parse_args() {
    VERBOSE=""
    ASSUME_YES=""
    WORK_DIR="$HOME/geti"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -v|--verbose)
                VERBOSE=1
                shift
                ;;
            -y|--yes)
                ASSUME_YES=1
                shift
                ;;
            -w|--work-dir)
                if [[ -z "${2:-}" ]]; then
                    echo "Error: --work-dir requires a path argument."
                    exit 1
                fi
                WORK_DIR="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "Error: unknown option '$1'"
                usage
                exit 1
                ;;
        esac
    done

    BUILD_TOOLS_DIR="$WORK_DIR/.build"
    UV_DIR="$BUILD_TOOLS_DIR/uv"
    NVM_DIR="$BUILD_TOOLS_DIR/nvm"
    LOG_FILE="$BUILD_TOOLS_DIR/.install.log"
}

confirm() {
    local prompt="$1"
    if [ -n "${ASSUME_YES:-}" ]; then
        return 0
    fi
    local response
    read -rp "$prompt [Y/n]: " response
    if [[ "$response" =~ ^[Nn]$ ]]; then
        return 1
    fi
}

run_cmd() {
    if [ -n "${VERBOSE:-}" ]; then
        "$@"
    else
        "$@" >>"$LOG_FILE" 2>&1
    fi
}

get_required_uv_version() {
    local version
    version=$(grep -A 3 '\[tool\.uv\]' "$WORK_DIR/application/backend/pyproject.toml" \
        | grep 'required-version' \
        | sed -E 's/.*"[^0-9]*([0-9]+\.[0-9]+\.[0-9]+).*/\1/' || true)
    if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Error: could not parse uv version from pyproject.toml" >&2
        return 1
    fi
    echo "$version"
}

get_required_node_version() {
    local version
    version=$(grep '"node"' "$WORK_DIR/application/ui/package.json" \
        | sed -E 's/.*">=v?([0-9]+\.[0-9]+\.[0-9]+)".*/\1/')
    if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Error: could not parse node version from package.json" >&2
        return 1
    fi
    echo "$version"
}

get_required_npm_version() {
    local version
    version=$(grep '"npm"' "$WORK_DIR/application/ui/package.json" \
        | sed -E 's/.*">=([0-9]+\.[0-9]+\.[0-9]+)".*/\1/')
    if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Error: could not parse npm version from package.json" >&2
        return 1
    fi
    echo "$version"
}

install_uv() {
    local uv_version
    uv_version=$(get_required_uv_version)

    if [ -x "$UV_DIR/uv" ]; then
        local installed_version
        installed_version=$("$UV_DIR/uv" --version | awk '{print $2}')
        if [ "$installed_version" = "$uv_version" ]; then
            echo "uv $uv_version found in $UV_DIR"
            return 0
        else
            echo "uv version mismatch: installed=$installed_version, required=$uv_version. Reinstalling..."
        fi
    fi

    echo "Installing uv $uv_version to: $UV_DIR"
    if ! confirm "Would you like to install uv now?"; then
        echo "uv installation skipped. Cannot continue without uv."
        exit 1
    fi

    if [ ! -d "$UV_DIR" ]; then
        mkdir -p "$UV_DIR"
    fi

    run_cmd bash -c "curl --proto '=https' --tlsv1.2 -LsSf 'https://github.com/astral-sh/uv/releases/download/${uv_version}/uv-installer.sh' | env UV_INSTALL_DIR='$UV_DIR' sh"
    echo "uv installation complete."
}

install_nvm() {
    export NVM_DIR

    if [ -s "$NVM_DIR/nvm.sh" ]; then
        source "$NVM_DIR/nvm.sh"
        echo "nvm found in $NVM_DIR."
        return 0
    fi

    echo "Installing nvm to: $NVM_DIR"
    if ! confirm "Would you like to install nvm now?"; then
        echo "nvm installation skipped. Cannot continue without nvm."
        exit 1
    fi

    if [ ! -d "$NVM_DIR" ]; then
        mkdir -p "$NVM_DIR"
    fi

    run_cmd bash -c "curl -sS -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash"
    source "$NVM_DIR/nvm.sh"
    echo "nvm installation complete."
}

install_npm() {
    local required_node_version required_npm_version
    required_node_version=$(get_required_node_version)
    required_npm_version=$(get_required_npm_version)

    NPM_BIN="$NVM_DIR/versions/node/v${required_node_version}/bin/npm"
    local node_bin installed_npm_version
    node_bin="$(dirname "$NPM_BIN")"

    if [ -x "$node_bin/node" ] && [ -x "$NPM_BIN" ]; then
        installed_npm_version=$("$NPM_BIN" --version)
        if [ "$(printf '%s\n' "$required_npm_version" "$installed_npm_version" | sort -V | head -n1)" = "$required_npm_version" ]; then
            echo "node $required_node_version and npm $installed_npm_version found in $node_bin."
            return 0
        fi

        echo "npm version too old: installed=$installed_npm_version, required>=$required_npm_version. Upgrading..."
        run_cmd "$NPM_BIN" install -g "npm@$required_npm_version"
        return 0
    fi

    echo "Required node $required_node_version not found in $NVM_DIR. Installing..."
    run_cmd nvm install "$required_node_version"

    installed_npm_version=$("$NPM_BIN" --version)
    if [ "$(printf '%s\n' "$required_npm_version" "$installed_npm_version" | sort -V | head -n1)" != "$required_npm_version" ]; then
        run_cmd "$NPM_BIN" install -g "npm@$required_npm_version"
    fi

    echo "node/npm installation complete: $node_bin"
}


detect_nvidia_gpus() {
    local gpu_count=0

    if command -v nvidia-smi &>/dev/null; then
        gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || true)
        if [ "$gpu_count" -gt 0 ]; then
            echo "Detected $gpu_count NVIDIA GPU(s) via nvidia-smi:"
            nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null
            return 0
        fi
    fi

    if [ -d /proc/driver/nvidia/gpus ]; then
        gpu_count=$(ls /proc/driver/nvidia/gpus 2>/dev/null | wc -l)
        if [ "$gpu_count" -gt 0 ]; then
            echo "Detected $gpu_count NVIDIA GPU(s) via /proc/driver/nvidia/gpus"
            return 0
        fi
    fi

    if command -v lspci &>/dev/null; then
        local gpus
        gpus=$(lspci | grep -i 'nvidia' | grep -i 'vga\|3d\|display' || true)
        if [ -n "$gpus" ]; then
            gpu_count=$(echo "$gpus" | wc -l)
            echo "Detected $gpu_count NVIDIA GPU(s) via lspci:"
            echo "$gpus"
            return 0
        fi
    fi

    echo "No NVIDIA GPUs detected."
    return 1
}

detect_intel_gpus() {
    local gpu_count=0

    if command -v xpu-smi &>/dev/null; then
        gpu_count=$(xpu-smi discovery 2>/dev/null | grep -c 'Device ID' || true)
        if [ "$gpu_count" -gt 0 ]; then
            echo "Detected $gpu_count Intel GPU(s) via xpu-smi:"
            xpu-smi discovery 2>/dev/null
            return 0
        fi
    fi

    if command -v sycl-ls &>/dev/null; then
        local intel_devs
        intel_devs=$(sycl-ls 2>/dev/null | grep -i 'intel' || true)
        if [ -n "$intel_devs" ]; then
            gpu_count=$(echo "$intel_devs" | wc -l)
            echo "Detected Intel GPU(s) via sycl-ls:"
            echo "$intel_devs"
            return 0
        fi
    fi

    if command -v lspci &>/dev/null; then
        local gpus
        gpus=$(lspci | grep -i 'intel' | grep -i 'vga\|3d\|display' || true)
        if [ -n "$gpus" ]; then
            gpu_count=$(echo "$gpus" | wc -l)
            echo "Detected $gpu_count Intel GPU(s) via lspci:"
            echo "$gpus"
            return 0
        fi
    fi

    echo "No Intel GPUs detected."
    return 1
}

preflight_checks() {
    if ! command -v git &>/dev/null; then
        echo "Error: git is not installed. Please install git and try again."
        exit 1
    fi

    if ! command -v curl &>/dev/null; then
        echo "Error: curl is not installed. Please install curl and try again."
        exit 1
    fi
}

ensure_source_code() {
    if [ ! -d "$WORK_DIR" ]; then
        echo "Cloning Intel Geti repository from $GIT_URL..."
        git -c advice.detachedHead=false clone --branch "$GIT_BRANCH" "$GIT_URL" "$WORK_DIR"
    else
        echo "Work directory $WORK_DIR already exists, skipping clone."
        local remote_url
        remote_url=$(git -C "$WORK_DIR" remote get-url origin 2>/dev/null)
        if [ "$remote_url" != "$GIT_URL" ]; then
            echo "Error: $WORK_DIR remote origin is '$remote_url', expected '$GIT_URL'."
            echo "Remove $WORK_DIR and re-run the installer."
            exit 1
        fi
        local current_sha expected_sha
        current_sha=$(git -C "$WORK_DIR" rev-parse HEAD 2>/dev/null)
        git -C "$WORK_DIR" fetch origin "$GIT_BRANCH" --tags 2>/dev/null || true
        expected_sha=$(git -C "$WORK_DIR" rev-parse "origin/$GIT_BRANCH" 2>/dev/null || git -C "$WORK_DIR" rev-parse "$GIT_BRANCH" 2>/dev/null || true)
        if [ "$current_sha" != "$expected_sha" ]; then
            echo "Switching to $GIT_BRANCH..."
            git -c advice.detachedHead=false -C "$WORK_DIR" checkout "$GIT_BRANCH"
        fi
    fi
}

install_build_tools() {
    install_uv
    install_nvm
    install_npm
}

detect_hardware() {
    HAS_NVIDIA_GPU=false
    HAS_INTEL_GPU=false

    if detect_nvidia_gpus; then
        HAS_NVIDIA_GPU=true
    fi

    if detect_intel_gpus; then
        HAS_INTEL_GPU=true
    fi

    if [ "$HAS_NVIDIA_GPU" = true ]; then
        ACCELERATOR="cuda"
    elif [ "$HAS_INTEL_GPU" = true ]; then
        ACCELERATOR="xpu"
    else
        ACCELERATOR="cpu"
    fi

    export ACCELERATOR
}

build_backend() {
    echo "Building venv using accelerator: $ACCELERATOR"
    cd "$WORK_DIR/application/backend"
    if [ -n "${VERBOSE:-}" ]; then
        "$UV_DIR/uv" sync --frozen --extra mqtt --extra "$ACCELERATOR"
    else
        "$UV_DIR/uv" sync --frozen --extra mqtt --extra "$ACCELERATOR" --quiet
    fi

    echo "Generating OpenAPI specification..."
    PYTHONPATH=. "$UV_DIR/uv" run --no-sync app/cli.py gen-api --target-path openapi.json
    cp openapi.json ../ui/src/api/openapi-spec.json
}

build_frontend() {
    cd "$WORK_DIR/application/ui"
    echo "Installing UI dependencies with npm..."
    export npm_config_yes=true
    run_cmd "$NPM_BIN" ci

    echo "Building API client with npm..."
    run_cmd "$NPM_BIN" run build:api

    echo "Building UI with npm..."
    run_cmd env ASSET_PREFIX="/html" "$NPM_BIN" run build
}

deploy_frontend() {
    local html_dir="$WORK_DIR/application/backend/html"

    echo "Copying built UI to backend html directory..."
    if [ -d "$html_dir" ]; then
      rm -rf "$html_dir"
    fi
    mkdir "$html_dir"
    cp -r "$WORK_DIR/application/ui/dist/"* "$html_dir"
}

register_shell_cmd() {
    local begin_marker="# BEGIN Intel Geti"
    local end_marker="# END Intel Geti"
    local shell_profile="${HOME}/.bashrc"

    if [ -n "${ZSH_VERSION:-}" ] || [[ "$SHELL" == */zsh ]]; then
        shell_profile="${HOME}/.zshrc"
    fi

    # Remove old marker block if present (idempotent update)
    if grep -qF "$begin_marker" "$shell_profile" 2>/dev/null; then
        sed -i "/$begin_marker/,/$end_marker/d" "$shell_profile"
    fi

    {
        echo ""
        echo "$begin_marker"
        echo "function geti { cd '$WORK_DIR/application/backend' && env STATIC_FILES_DIR=html \"\$@\" '$UV_DIR/uv' run app/main.py; }"
        echo "$end_marker"
    } >> "$shell_profile"

    echo "Function 'geti' written to $shell_profile"
    echo "Run 'source $shell_profile' to activate it in the current session."
    echo "Example: geti HOST=0.0.0.0 PORT=8080"
}

main() {
    parse_args "$@"

    preflight_checks
    ensure_source_code

    # Initialize log file and build tools directory
    mkdir -p "$BUILD_TOOLS_DIR"
    : > "$LOG_FILE"

    install_build_tools
    detect_hardware
    build_backend
    build_frontend
    deploy_frontend
    register_shell_cmd
}

main "$@"
