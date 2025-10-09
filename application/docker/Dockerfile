# syntax=docker/dockerfile:1.7

#######
# Backend base
#######
FROM python:3.13-slim@sha256:087a9f3b880e8b2c7688debb9df2a5106e060225ebd18c264d5f1d7a73399db0 AS backend-base
COPY --from=docker.io/astral/uv:0.8.8@sha256:67b2bcccdc103d608727d1b577e58008ef810f751ed324715eb60b3f0c040d30 /uv /uvx /bin/

# Set working directory
WORKDIR /app

ENV PYTHONPATH=/app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1=1.7.0-1+b2 \
    libglx-mesa0=25.0.7-2 \
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
FROM node:24-alpine3.22@sha256:f8baf2c963e3bff767993135ae3447bb433e8d64a5e4bb65780cd29ef3a525c2 AS build_rest_api_specs

WORKDIR /home/app/web_ui/

COPY --link ui/package.json .
COPY --link ui/openapi.json /home/app/web_ui/src/api/openapi-spec.json
RUN npm run build:api

#############
# geti_ui_packages
#############
FROM node:24-alpine3.22@sha256:f8baf2c963e3bff767993135ae3447bb433e8d64a5e4bb65780cd29ef3a525c2 AS geti_ui_packags

WORKDIR /home/app/web_ui/
RUN apk add --no-cache git=2.49.1-r0
COPY --link ./ui/package.json ./
RUN npm run clone-geti-ui-packages

#############
# web_ui_deps
#############
FROM node:24-alpine3.22@sha256:f8baf2c963e3bff767993135ae3447bb433e8d64a5e4bb65780cd29ef3a525c2 AS base_web_ui

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
FROM backend-base AS geti-tune

# Install nginx
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx=1.26.* \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=web_ui /home/app/web_ui/dist/ /usr/share/nginx/html

EXPOSE 80

CMD ["sh", "-c", "nginx && exec uv run ./app/main.py;"]
