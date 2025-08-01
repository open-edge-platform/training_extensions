#######
# Base
#######
FROM python:3.10-slim@sha256:034724ef64585eeb0e82385e9aabcbeabfe5f7cae2c2dcedb1da95114372b6d7 AS base
COPY --from=ghcr.io/astral-sh/uv:0.7.14@sha256:cda0fdc9b6066975ba4c791597870d18bc3a441dfc18ab24c5e888c16e15780c /uv /uvx /bin/

# Set working directory
WORKDIR /app

ENV PYTHONPATH=/app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project

# Copy project files
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable

COPY app ./app

#######
# Server
#######
FROM base AS server

# Expose port
EXPOSE 7860

# Set the entry point
CMD ["uv", "run", "./app/main.py"]
