# syntax=docker/dockerfile:1.7

##############
# Backend base
##############
FROM python:3.13-slim@sha256:85dfbf1b566b7addfe645faea9938e81a0a01a83580b0ea05fb23706357d77fb AS backend-base
COPY --from=docker.io/astral/uv:0.9.7@sha256:ba4857bf2a068e9bc0e64eed8563b065908a4cd6bfb66b531a9c424c8e25e142 /uv /uvx /bin/

# Set working directory
WORKDIR /application

# Install required build dependencies
# TODO remove 'git' once all dependencies are available on PyPI
RUN apt-get update  \
    && apt-get install -y --no-install-recommends \
        build-essential=12.12 \
        curl=8.14.1-2 \
        git=1:2.47.3-0+deb13u1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Rust compiler
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=backend/pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-editable --extra mqtt

COPY backend/app ./app

##################
# Geti UI packages
##################
FROM node:24-alpine3.22@sha256:3e843c608bb5232f39ecb2b25e41214b958b0795914707374c8acc28487dea17 AS geti-ui-packages

WORKDIR /home/app/web_ui/
RUN apk add --no-cache git=2.49.1-r0
COPY --link ./ui/package.json ./
RUN npm run clone-geti-ui-packages

#############
# Web UI deps
#############
FROM node:24-alpine3.22@sha256:3e843c608bb5232f39ecb2b25e41214b958b0795914707374c8acc28487dea17 AS web-ui-base

WORKDIR /home/app/web_ui/

COPY --link ui/package.json ./
COPY --link ui/package-lock.json ./
COPY --from=geti-ui-packages /home/app/web_ui/packages/ /home/app/web_ui/packages/
RUN npm ci --audit=false --ignore-scripts

COPY --link ui/tsconfig.json ./
COPY --link ui/rsbuild.config.ts ./
COPY --link ui/src/ src/

ARG PUBLIC_API_BASE_URL=""
ENV PUBLIC_API_BASE_URL ${PUBLIC_API_BASE_URL}

########
# Web UI
########
FROM web-ui-base as web-ui
RUN npm run build

######################
# Server (CPU version)
######################
FROM python:3.13-slim@sha256:58c30f5bfaa718b5803a53393190b9c68bd517c44c6c94c1b6c8c172bcfad040 AS geti-tune-cpu

WORKDIR /application
ENV PYTHONPATH=/application

# Install nginx & runtime system dependencies
# TODO remove 'git' once all dependencies are available on PyPI
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git=1:2.47.3-0+deb13u1 \
        libgl1=1.7.0-1+b2 \
        libglib2.0-0=2.84.4-3~deb13u1 \
        libglx-mesa0=25.0.7-2 \
        nginx=1.26.* \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=web-ui /home/app/web_ui/dist/ /usr/share/nginx/html
COPY --from=backend-base /application /application
COPY --from=backend-base /bin/uv /bin/uv

# Install Python dependencies for CPU version
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=backend/pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-editable --inexact --extra cpu

EXPOSE 80

CMD ["sh", "-c", "nginx && exec uv run ./app/main.py;"]

######################
# Server (GPU version)
######################
FROM python:3.13-slim@sha256:58c30f5bfaa718b5803a53393190b9c68bd517c44c6c94c1b6c8c172bcfad040 AS geti-tune-gpu

WORKDIR /application
ENV PYTHONPATH=/application

RUN apt-get update  \
    && apt-get install -y --no-install-recommends wget=1.25.0-2 \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm -rf cuda-keyring_1.1-1_all.deb \
    && apt-get remove -y wget

# Install nginx & runtime system dependencies, including the Nvidia drivers.
# Note that CUDA runtime dependencies are shipped through PyTorch, so there's no need to install the CUDA toolkit here.
# TODO remove 'git' once all dependencies are available on PyPI
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git=1:2.47.3-0+deb13u1 \
        libgl1=1.7.0-1+b2 \
        libglib2.0-0=2.84.4-3~deb13u1 \
        libglx-mesa0=25.0.7-2 \
        nvidia-open \
        nginx=1.26.* \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=web-ui /home/app/web_ui/dist/ /usr/share/nginx/html
COPY --from=backend-base /application /application
COPY --from=backend-base /bin/uv /bin/uv

# Install Python dependencies for GPU version
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=backend/pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-editable --inexact --extra cuda

EXPOSE 80

CMD ["sh", "-c", "nginx && exec uv run ./app/main.py;"]

######################
# Server (XPU version)
######################
FROM python:3.13-slim@sha256:58c30f5bfaa718b5803a53393190b9c68bd517c44c6c94c1b6c8c172bcfad040 AS geti-tune-xpu

WORKDIR /application
ENV PYTHONPATH=/application

# Install nginx and runtime system dependencies.
# Includes Intel graphics drivers (steps from https://dgpu-docs.intel.com/driver/client/overview.html#ubuntu-latest)
# TODO remove 'git' once all dependencies are available on PyPI
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN echo "deb [arch=amd64 trusted=yes] https://ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu noble main" | tee /etc/apt/sources.list.d/kobuk-intel.list
# hadolint ignore=DL3008
RUN apt-get update  \
    && apt-get install -y --no-install-recommends \
        git=1:2.47.3-0+deb13u1 \
        libgl1=1.7.0-1+b2 \
        libglib2.0-0=2.84.4-3~deb13u1 \
        libglx-mesa0=25.0.7-2 \
        libze-intel-gpu1 libze1 libze-dev intel-metrics-discovery intel-opencl-icd clinfo intel-gsc intel-ocloc \
        intel-media-va-driver-non-free libmfx-gen1 libvpl2 libvpl-tools libva-glx2 va-driver-all vainfo \
        nginx=1.26.* \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=web-ui /home/app/web_ui/dist/ /usr/share/nginx/html
COPY --from=backend-base /application /application
COPY --from=backend-base /bin/uv /bin/uv

# Install Python dependencies for XPU version
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=backend/pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-editable --inexact --extra xpu

EXPOSE 80

CMD ["sh", "-c", "nginx && exec uv run ./app/main.py;"]
