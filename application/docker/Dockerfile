# syntax=docker/dockerfile:1.7

#######
# Backend base
#######
FROM python:3.13-slim@sha256:58c30f5bfaa718b5803a53393190b9c68bd517c44c6c94c1b6c8c172bcfad040 AS backend-build
COPY --from=docker.io/astral/uv:0.8.8@sha256:67b2bcccdc103d608727d1b577e58008ef810f751ed324715eb60b3f0c040d30 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Install required build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git=1:2.47.3-0+deb13u1 \
    build-essential=12.12 \
    curl=8.14.1-2 \
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
    uv sync --frozen --no-editable

COPY backend/app ./app

#############
# build_rest_api_specs
#############
FROM node:24-alpine3.22@sha256:3e843c608bb5232f39ecb2b25e41214b958b0795914707374c8acc28487dea17 AS build_rest_api_specs

WORKDIR /home/app/web_ui/

COPY --link ui/package.json .
COPY --link ui/openapi.json /home/app/web_ui/src/api/openapi-spec.json
RUN npm run build:api

#############
# geti_ui_packages
#############
FROM node:24-alpine3.22@sha256:3e843c608bb5232f39ecb2b25e41214b958b0795914707374c8acc28487dea17 AS geti_ui_packags

WORKDIR /home/app/web_ui/
RUN apk add --no-cache git=2.49.1-r0
COPY --link ./ui/package.json ./
RUN npm run clone-geti-ui-packages

#############
# web_ui_deps
#############
FROM node:24-alpine3.22@sha256:3e843c608bb5232f39ecb2b25e41214b958b0795914707374c8acc28487dea17 AS base_web_ui

WORKDIR /home/app/web_ui/

COPY --link ui/package.json ./
COPY --link ui/package-lock.json ./
COPY --from=geti_ui_packags /home/app/web_ui/packages/ /home/app/web_ui/packages/
RUN npm ci --audit=false --ignore-scripts

COPY --link ui/tsconfig.json ./
COPY --link ui/rsbuild.config.ts ./
COPY --link ui/src/ src/

ARG PUBLIC_API_BASE_URL=""
ENV PUBLIC_API_BASE_URL ${PUBLIC_API_BASE_URL}

#############
# web_ui build
#############
FROM base_web_ui as web_ui
RUN npm run build

#############
# Server
#############
FROM python:3.13-slim@sha256:58c30f5bfaa718b5803a53393190b9c68bd517c44c6c94c1b6c8c172bcfad040 AS geti-tune

WORKDIR /app
ENV PYTHONPATH=/app

# Install nginx & OpenCV required libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1=1.7.0-1+b2 \
    libglx-mesa0=25.0.7-2 \
    nginx=1.26.* \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=web_ui /home/app/web_ui/dist/ /usr/share/nginx/html
COPY --from=backend-build /app /app
COPY --from=backend-build /bin/uv /bin/uv

EXPOSE 80

CMD ["sh", "-c", "nginx && exec uv run ./app/main.py;"]
