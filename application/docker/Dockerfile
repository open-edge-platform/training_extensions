# syntax=docker/dockerfile:1.7

##############
# Backend base
##############
FROM python:3.13-slim@sha256:baf66684c5fcafbda38a54b227ee30ec41e40af1e4073edee3a7110a417756ba AS backend-base

# uv / uvx
COPY --from=docker.io/astral/uv:0.9.7@sha256:ba4857bf2a068e9bc0e64eed8563b065908a4cd6bfb66b531a9c424c8e25e142 /uv /uvx /bin/

WORKDIR /application/backend

# Build deps (for compiling / Rust-based crates)
# TODO: remove 'git' once all dependencies are on PyPI
RUN apt-get update  \
    && apt-get install -y --no-install-recommends \
        build-essential=12.12 \
        curl=8.14.1-2* \
        git=1:2.47.3-0* \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Rust toolchain
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

##########
# CPU Base
##########
FROM backend-base AS cpu-base

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=application/backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=application/backend/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=library,target=/library,rw \
    uv sync --frozen --no-dev --no-editable --extra mqtt --extra cpu

##########
# GPU Base
##########
FROM backend-base AS gpu-base

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=application/backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=application/backend/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=library,target=/library,rw \
    uv sync --frozen --no-dev --no-editable --extra mqtt --extra cuda

##########
# XPU Base
##########
FROM backend-base AS xpu-base

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=application/backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=application/backend/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=library,target=/library,rw \
    uv sync --frozen --no-dev --no-editable --extra mqtt --extra xpu

##################
# Geti UI packages
##################
FROM node:24-alpine3.22@sha256:4f4a059445c5a6ef2b9d169d9afde176301263178141fc05ba657dab1c84f9a7 AS geti-ui-packages

WORKDIR /home/application/ui/
RUN apk add --no-cache git=2.49.1-r0
COPY --link application/ui/package.json ./
RUN npm run clone-geti-ui-packages

#############
# Web UI deps
#############
FROM node:24-alpine3.22@sha256:4f4a059445c5a6ef2b9d169d9afde176301263178141fc05ba657dab1c84f9a7 AS web-ui-base

WORKDIR /home/application/ui/

COPY --link application/ui/package.json ./
COPY --link application/ui/package-lock.json ./
COPY --from=geti-ui-packages /home/application/ui/packages/ /home/application/ui/packages/
RUN npm ci --audit=false --ignore-scripts

COPY --link application/ui/tsconfig.json ./
COPY --link application/ui/rsbuild.config.ts ./
COPY --link application/ui/src/ src/

########
# Web UI
########
FROM web-ui-base AS web-ui

ARG ASSET_PREFIX=""
ENV ASSET_PREFIX=${ASSET_PREFIX}
ARG PUBLIC_API_BASE_URL=""
ENV PUBLIC_API_BASE_URL=${PUBLIC_API_BASE_URL}

RUN npm run build

#########################
# Base for CPU, GPU and XPU runtime images
#########################
FROM python:3.13-slim@sha256:baf66684c5fcafbda38a54b227ee30ec41e40af1e4073edee3a7110a417756ba AS runtime-base

ARG STATIC_FILES_DIR=""
ENV STATIC_FILES_DIR=${STATIC_FILES_DIR}
ARG UID=10001
ARG USER_NAME="non-root"

# Create secure non-root user
RUN useradd -l -u ${UID} ${USER_NAME}

# Common system deps (CPU/GPU/XPU all need these)
# TODO: remove 'git' once all dependencies are available on PyPI
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git=1:2.47.3-0* \
        libgl1=1.7.0-1+b2 \
        libglib2.0-0t64=2.84.4-3~deb13u2 \
        libglx-mesa0=25.0.7-2 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Backend code + uv from backend-base
COPY --link application/backend/app /application/backend/app
COPY --from=backend-base /bin/uv /bin/uv
COPY --from=web-ui /home/application/ui/dist/ ${STATIC_FILES_DIR}

# Make runtime directories writable for non-root user
RUN mkdir -p /home/${USER_NAME}/.cache/uv && \
    chown -R ${UID}:${UID} \
    /application \
    /home/${USER_NAME}

EXPOSE 7860

WORKDIR /application/backend
ENV PYTHONPATH=/application/backend

USER ${USER_NAME}

CMD ["uv", "run", "app/main.py"]

######################
# Server (CPU version)
######################
FROM runtime-base AS geti-tune-cpu

COPY --from=cpu-base --chown=${UID}:${UID} /application/backend/.venv /application/backend/.venv

######################
# Server (GPU version)
######################
FROM runtime-base AS geti-tune-gpu

USER root

RUN apt-get update  \
    && apt-get install -y --no-install-recommends wget=1.25.0-2 \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm -rf cuda-keyring_1.1-1_all.deb \
    && apt-get remove -y wget

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      nvidia-open=580.105.08-1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

USER ${USER_NAME}

COPY --from=gpu-base --chown=${UID}:${UID} /application/backend/.venv /application/backend/.venv

#######################
## Server (XPU version)
#######################
FROM runtime-base AS geti-tune-xpu

USER root

# XPU / Intel stack
# Includes Intel graphics drivers (steps from https://dgpu-docs.intel.com/driver/client/overview.html#ubuntu-latest)
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN echo "deb [arch=amd64 trusted=yes] https://ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu noble main" \
    | tee /etc/apt/sources.list.d/kobuk-intel.list

# hadolint ignore=DL3008
RUN apt-get update  \
    && apt-get install -y --no-install-recommends \
        libze-intel-gpu1 libze1 libze-dev intel-metrics-discovery intel-opencl-icd clinfo intel-gsc intel-ocloc \
        intel-media-va-driver-non-free libmfx-gen1 libvpl2 libvpl-tools libva-glx2 va-driver-all vainfo \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

USER ${USER_NAME}

COPY --from=xpu-base --chown=${UID}:${UID} /application/backend/.venv /application/backend/.venv
