# syntax=docker/dockerfile:1.7

##############
# Backend base
##############
FROM python:3.13-slim@sha256:d168b8d9eb761f4d3fe305ebd04aeb7e7f2de0297cec5fb2f8f6403244621664 AS backend-base

# uv / uvx
COPY --from=docker.io/astral/uv:0.10.4@sha256:4cac394b6b72846f8a85a7a0e577c6d61d4e17fe2ccee65d9451a8b3c9efb4ac /uv /uvx /bin/

# Increase HTTP timeout for large packages (e.g. torch)
ENV UV_HTTP_TIMEOUT=300
# Compile bytecode at install time for faster container startup
ENV UV_COMPILE_BYTECODE=1
# Use copy instead of hardlinks (cache mount is on a different filesystem)
ENV UV_LINK_MODE=copy
# Don't download a Python interpreter; use the one from the base image
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /application/backend

# Build deps (for compiling / Rust-based crates)
# TODO: remove 'git' once all dependencies are on PyPI
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update  \
    && apt-get install -y --no-install-recommends \
        build-essential=12.12 \
        curl=8.14.1-2* \
        git=1:2.47.3-0*

# Install Rust toolchain
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

##########
# CPU Base
##########
FROM backend-base AS cpu-base

# Step 1: Install all third-party dependencies (cache key = uv.lock + pyproject.toml + version only).
# --no-install-project skips building/installing the local project and its path deps (getitune),
# so changes to library/ source code do NOT invalidate this (slow) layer.
# Note: __init__.py is needed because getitune uses dynamic versioning (attr = "getitune.__version__").
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=application/backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=application/backend/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=application/VERSION,target=../VERSION \
    --mount=type=bind,source=library/pyproject.toml,target=/library/pyproject.toml \
    --mount=type=bind,source=library/src/getitune/__init__.py,target=/library/src/getitune/__init__.py \
    uv sync --frozen --no-dev --no-editable --no-install-project --no-install-workspace --extra mqtt --extra cpu

# Step 2: Build and install the local getitune package + the project itself.
# This layer is invalidated by library source changes, but is fast since all
# third-party deps are already installed from step 1.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=application/backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=application/backend/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=application/VERSION,target=../VERSION \
    --mount=type=bind,source=library,target=/library,rw \
    uv sync --frozen --no-dev --no-editable --extra mqtt --extra cpu

###########
# CUDA Base
###########
FROM backend-base AS cuda-base

# Step 1: Install all third-party dependencies (cache key = uv.lock + pyproject.toml + version only).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=application/backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=application/backend/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=application/VERSION,target=../VERSION \
    --mount=type=bind,source=library/pyproject.toml,target=/library/pyproject.toml \
    --mount=type=bind,source=library/src/getitune/__init__.py,target=/library/src/getitune/__init__.py \
    uv sync --frozen --no-dev --no-editable --no-install-project --no-install-workspace --extra mqtt --extra cuda

# Step 2: Build and install the local getitune package + the project itself.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=application/backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=application/backend/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=application/VERSION,target=../VERSION \
    --mount=type=bind,source=library,target=/library,rw \
    uv sync --frozen --no-dev --no-editable --extra mqtt --extra cuda

##########
# XPU Base
##########
FROM backend-base AS xpu-base

# Step 1: Install all third-party dependencies (cache key = uv.lock + pyproject.toml + version only).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=application/backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=application/backend/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=application/VERSION,target=../VERSION \
    --mount=type=bind,source=library/pyproject.toml,target=/library/pyproject.toml \
    --mount=type=bind,source=library/src/getitune/__init__.py,target=/library/src/getitune/__init__.py \
    uv sync --frozen --no-dev --no-editable --no-install-project --no-install-workspace --extra mqtt --extra xpu

# Step 2: Build and install the local getitune package + the project itself.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=application/backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=application/backend/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=application/VERSION,target=../VERSION \
    --mount=type=bind,source=library,target=/library,rw \
    uv sync --frozen --no-dev --no-editable --extra mqtt --extra xpu

##################
# Geti UI packages
##################
FROM node:24-alpine3.22@sha256:4f33c7804fa7775b87875f512e25e515692210bfdeafd715eb8bc441c364e2f6 AS geti-ui-packages

WORKDIR /home/application/ui/
RUN apk add --no-cache git=2.49.1-r0
COPY --link application/ui/package.json ./
RUN npm run clone-geti-ui-packages

#############
# Web UI deps
#############
FROM node:24-alpine3.22@sha256:4f33c7804fa7775b87875f512e25e515692210bfdeafd715eb8bc441c364e2f6 AS web-ui-base

WORKDIR /home/application/ui/

COPY --link application/ui/package.json ./
COPY --link application/ui/package-lock.json ./
COPY --from=geti-ui-packages /home/application/ui/packages/ /home/application/ui/packages/
RUN npm ci --audit=false --ignore-scripts

COPY --link application/ui/tsconfig.json ./
COPY --link application/ui/rsbuild.config.ts ./
COPY --link application/ui/src/ src/
COPY --link application/ui/public/ public/

########
# Web UI
########
FROM web-ui-base AS web-ui

ENV ASSET_PREFIX="/html"

RUN npm run build

#########################
# Base for CPU, GPU and XPU runtime images
#########################
FROM python:3.13-slim@sha256:d168b8d9eb761f4d3fe305ebd04aeb7e7f2de0297cec5fb2f8f6403244621664 AS runtime-base

LABEL org.opencontainers.image.source="https://github.com/open-edge-platform/training_extensions"
LABEL org.opencontainers.image.description="Geti – train, optimize and deploy deep-learning models"
LABEL org.opencontainers.image.revision="${GIT_SHA}"

ARG UID=10001
ARG USER_NAME="non-root"

ENV STATIC_FILES_DIR="/application/html"
ENV DATA_DIR="/application/data"
ENV LOG_DIR="/application/logs"
ENV PYTHONPATH=/application/backend

# Create secure non-root user, add it to the 'video' group (for /dev/videoN access),
# and remove unused package managers to reduce image size
RUN groupadd -f video && \
    useradd -l -u ${UID} -G video ${USER_NAME} && \
    pip uninstall -y setuptools pip wheel && \
    rm -rf /root/.cache/pip

# Create directory and set ownership so mounting a volume works correctly
RUN mkdir -p ${DATA_DIR} && chown ${UID}:${UID} ${DATA_DIR}
RUN mkdir -p ${LOG_DIR} && chown ${UID}:${UID} ${LOG_DIR}

# Common system deps (CPU/GPU/XPU all need these)
# TODO: remove 'git' once all dependencies are available on PyPI
# libv4l-dev: V4L2 userspace libraries required by OpenCV to capture from USB cameras (/dev/videoN)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        git=1:2.47.3-0* \
        libgl1=1.7.0-1+b2 \
        libglib2.0-0t64=2.84.4-3~deb13u2 \
        libglx-mesa0=25.0.7-2 \
        libv4l-0t64=1.30.1-1

# Place the venv at the front of PATH so we can call python directly (no uv needed at runtime)
ENV PATH="/application/backend/.venv/bin:${PATH}"

# Create directories and set ownership for the non-root user.
RUN mkdir -p /application/backend/app ${STATIC_FILES_DIR} /home/${USER_NAME} && \
    chown -R ${UID}:${UID} \
    /application \
    /home/${USER_NAME}

WORKDIR /application/backend

USER ${USER_NAME}

CMD ["python", "app/main.py"]

######################
# Server (CPU version)
######################
FROM runtime-base AS geti-cpu

COPY --from=cpu-base --chown=${UID}:${UID} /application/backend/.venv /application/backend/.venv

# App code + UI assets last – source-only changes only invalidate this layer
COPY --link application/backend/app /application/backend/app
COPY --link --from=web-ui /home/application/ui/dist/ ${STATIC_FILES_DIR}

######################
# Server (CUDA version)
######################
FROM runtime-base AS geti-cuda

COPY --from=cuda-base --chown=${UID}:${UID} /application/backend/.venv /application/backend/.venv

# App code + UI assets last – source-only changes only invalidate this layer
COPY --link application/backend/app /application/backend/app
COPY --link --from=web-ui /home/application/ui/dist/ ${STATIC_FILES_DIR}

#######################
## Server (XPU version)
#######################
FROM runtime-base AS geti-xpu

USER root

# XPU / Intel stack
# Includes Intel graphics drivers (steps from https://dgpu-docs.intel.com/driver/client/overview.html#ubuntu-latest)
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN echo "deb [arch=amd64 trusted=yes] https://ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu noble main" \
    | tee /etc/apt/sources.list.d/kobuk-intel.list

# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update  \
    && apt-get install -y --no-install-recommends \
        libze-intel-gpu1 libze1 libze-dev intel-metrics-discovery intel-opencl-icd clinfo intel-gsc intel-ocloc \
        intel-media-va-driver-non-free libmfx-gen1 libvpl2 libvpl-tools libva-glx2 va-driver-all vainfo

USER ${USER_NAME}

COPY --from=xpu-base --chown=${UID}:${UID} /application/backend/.venv /application/backend/.venv

# App code + UI assets last – source-only changes only invalidate this layer
COPY --link application/backend/app /application/backend/app
COPY --link --from=web-ui /home/application/ui/dist/ ${STATIC_FILES_DIR}

