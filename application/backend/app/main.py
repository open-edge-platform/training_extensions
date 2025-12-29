# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# export no_proxy="localhost, 127.0.0.1, ::1"
# Start with:
#  - uv run app/main.py
# or use docker
#  - docker compose up

import importlib
import logging
import pkgutil
from collections.abc import Awaitable, Callable
from os import getenv
from pathlib import Path
from typing import cast

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.api import routers
from app.core.logging import InterceptHandler
from app.lifecycle import lifespan
from app.settings import get_settings

settings = get_settings()
logging.basicConfig(handlers=[InterceptHandler()], level=settings.log_level, force=True)
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=settings.description,
    openapi_url=settings.openapi_url,
    redoc_url=None,
    docs_url=None,
    lifespan=lifespan,
    # TODO add contact info
    # TODO add license
)

app.add_middleware(  # TODO restrict settings in production
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all API routers from the routers package
for router_info in pkgutil.iter_modules(routers.__path__):
    router_name = router_info.name
    module = importlib.import_module(f"app.api.routers.{router_name}")
    if hasattr(module, "router"):
        app.include_router(module.router)

cur_dir = Path(__file__).parent


@app.get("/api/docs", include_in_schema=False)
async def get_scalar_docs() -> FileResponse:
    """Shows docs for our OpenAPI specification using scalar"""
    return FileResponse(cur_dir / "static" / "scalar.html")


@app.get("/stream", include_in_schema=False)
async def get_webrtc_stream() -> FileResponse:
    """Get webrtc player"""
    return FileResponse(cur_dir / "static" / "webrtc.html")


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint"""
    return {"status": "ok"}


@app.middleware("http")
async def security_headers_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Add COEP and COOP security headers to all HTTP responses."""
    response = await call_next(request)
    response.headers.setdefault("Cross-Origin-Embedder-Policy", "require-corp")
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    return response


static_dir = settings.static_files_dir
if static_dir is not None and static_dir.is_dir() and any(static_dir.iterdir()):
    asset_prefix = getenv("ASSET_PREFIX", "/html")
    logger.info("Serving static files from {} by context {}", static_dir, asset_prefix)

    app.mount(asset_prefix, StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa() -> FileResponse:
        """Serve the Single Page Application (SPA) index.html file for any path."""
        return FileResponse(cast(Path, static_dir) / "index.html")


def main() -> None:
    """Main application entry point"""
    logger.info(f"Starting {settings.app_name} in {settings.environment} mode")
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        # FIXME: reload mode currently does not work with multiple workers
        # reload=settings.environment == "dev",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
