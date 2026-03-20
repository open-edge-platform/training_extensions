# syntax=docker/dockerfile:1.7

##############
# Backend base
##############
FROM python:3.13-slim@sha256:3de9a8d7aedbb7984dc18f2dff178a7850f16c1ae7c34ba9d7ecc23d0755e35f AS backend-base

# uv / uvx
COPY --from=docker.io/astral/uv:0.10.4@sha256:4cac394b6b72846f8a85a7a0e577c6d61d4e17fe2ccee65d9451a8b3c9efb4ac /uv /uvx /bin/

ENV UV_HTTP_TIMEOUT=300

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

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=application/backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=application/backend/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=library,target=/library,rw \
    uv sync --frozen --no-dev --no-editable --extra mqtt --extra cpu

###########
# CUDA Base
###########
FROM backend-base AS cuda-base

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
FROM node:24-alpine3.22@sha256:bb089be859f2741e5ede9d85f47dc7daca754015b50e9642a7a45c5252807d2c AS geti-ui-packages

WORKDIR /home/application/ui/
RUN apk add --no-cache git=2.49.1-r0
COPY --link application/ui/package.json ./
RUN npm run clone-geti-ui-packages

#############
# Web UI deps
#############
FROM node:24-alpine3.22@sha256:bb089be859f2741e5ede9d85f47dc7daca754015b50e9642a7a45c5252807d2c AS web-ui-base

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
FROM python:3.13-slim@sha256:3de9a8d7aedbb7984dc18f2dff178a7850f16c1ae7c34ba9d7ecc23d0755e35f AS runtime-base

ARG UID=10001
ARG USER_NAME="non-root"

ENV STATIC_FILES_DIR="/application/html"
ENV DATA_DIR="/application/data"
ENV LOG_DIR="/application/logs"
ENV PYTHONPATH=/application/backend

# Create secure non-root user
RUN useradd -l -u ${UID} ${USER_NAME}

# Create directory and set ownership so mounting a volume works correctly
RUN mkdir -p ${DATA_DIR} && chown ${UID}:${UID} ${DATA_DIR}
RUN mkdir -p ${LOG_DIR} && chown ${UID}:${UID} ${LOG_DIR}

# Common system deps (CPU/GPU/XPU all need these)
# TODO: remove 'git' once all dependencies are available on PyPI
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        git=1:2.47.3-0* \
        libgl1=1.7.0-1+b2 \
        libglib2.0-0t64=2.84.4-3~deb13u2 \
        libglx-mesa0=25.0.7-2

# Backend code + uv from backend-base
COPY --link application/backend/app /application/backend/app
COPY --from=backend-base /bin/uv /bin/uv
COPY --from=web-ui /home/application/ui/dist/ ${STATIC_FILES_DIR}

# Make runtime directories writable for non-root user
RUN mkdir -p /home/${USER_NAME}/.cache/uv && \
    chown -R ${UID}:${UID} \
    /application \
    /home/${USER_NAME}

WORKDIR /application/backend

USER ${USER_NAME}

CMD ["uv", "run", "app/main.py"]

######################
# Server (CPU version)
######################
FROM runtime-base AS geti-cpu

COPY --from=cpu-base --chown=${UID}:${UID} /application/backend/.venv /application/backend/.venv

######################
# Server (CUDA version)
######################
FROM runtime-base AS geti-cuda

USER root

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update  \
    && apt-get install -y --no-install-recommends wget=1.25.0-2 \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm -rf cuda-keyring_1.1-1_all.deb \
    && apt-get remove -y wget

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
      nvidia-open=590.48.01-1

USER ${USER_NAME}

COPY --from=cuda-base --chown=${UID}:${UID} /application/backend/.venv /application/backend/.venv

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
