# syntax=docker/dockerfile:1.7

#######
# Backend base
#######
FROM python:3.13-slim@sha256:6f79e7a10bb7d0b0a50534a70ebc78823f941fba26143ecd7e6c5dca9d7d7e8a AS backend-base
COPY --from=docker.io/astral/uv:0.8.8@sha256:67b2bcccdc103d608727d1b577e58008ef810f751ed324715eb60b3f0c040d30 /uv /uvx /bin/

# Set working directory
WORKDIR /app

ENV PYTHONPATH=/app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx=22.3.6-1+deb12u1 \
    libglib2.0-0=2.74.* \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=backend/pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-editable

COPY backend/app ./app

#############
# build_rest_api_specs
#############
FROM node:24-alpine3.22@sha256:7aaba6b13a55a1d78411a1162c1994428ed039c6bbef7b1d9859c25ada1d7cc5 AS build_rest_api_specs

WORKDIR /home/app/web_ui/

COPY --link ui/package.json .
COPY --link ui/openapi.json /home/app/web_ui/src/api/openapi-spec.json
RUN npm run build:api

#############
# geti_ui_packages
#############
FROM node:24-alpine3.22@sha256:7aaba6b13a55a1d78411a1162c1994428ed039c6bbef7b1d9859c25ada1d7cc5 AS geti_ui_packags

WORKDIR /home/app/web_ui/
RUN apk add --no-cache git=2.49.1-r0
COPY --link ./ui/package.json ./
RUN npm run clone-geti-ui-packages

#############
# web_ui_deps
#############
FROM node:24-alpine3.22@sha256:7aaba6b13a55a1d78411a1162c1994428ed039c6bbef7b1d9859c25ada1d7cc5 AS base_web_ui

WORKDIR /home/app/web_ui/

COPY --link ui/package.json ./
COPY --link ui/package-lock.json ./
COPY --from=geti_ui_packags /home/app/web_ui/packages/ /home/app/web_ui/packages/
RUN npm ci --audit=false --ignore-scripts

COPY --link ui/eslint.config.js ./
COPY --link ui/.prettierrc ./
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
FROM backend-base AS geti-tune

# Install nginx
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx=1.22.* \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=web_ui /home/app/web_ui/dist/ /usr/share/nginx/html

EXPOSE 80

CMD ["sh", "-c", "nginx; uv run ./app/main.py;"]
