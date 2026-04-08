# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# export no_proxy="localhost, 127.0.0.1, ::1"
# Start with:
#  - uv run app/main.py
# or use docker
#  - docker compose up

import sys

if getattr(sys, "frozen", False) and __name__ == "__main__":
    print("Calling multiprocessing.freeze_support()")
    import multiprocessing

    # Pyinstaller requires this method to be called in "frozen" applications if multiprocessing module is
    # used to prevent issues. This line must be called before any attempt to use multiprocessing module, so it makes
    # sense to put it in the very beginning.
    # https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html#multi-processing
    multiprocessing.freeze_support()

import logging
from collections.abc import Awaitable, Callable
from os import getenv
from pathlib import Path
from typing import Annotated, cast

import uvicorn
from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.api.dependencies import get_license_service
from app.api.routers import (
    dataset_ie,
    dataset_revisions,
    datasets,
    jobs,
    license,
    media,
    model_architectures,
    models,
    pipelines,
    projects,
    sinks,
    sources,
    system,
    training_configurations,
    webrtc,
)
from app.core.logging import InterceptHandler
from app.lifecycle import lifespan
from app.services.base import ResourceNotFoundError
from app.services.license_service import LicenseService
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
app.include_router(dataset_ie.router)
app.include_router(dataset_revisions.router)
app.include_router(datasets.router)
app.include_router(jobs.router)
app.include_router(license.router)
app.include_router(media.router)
app.include_router(model_architectures.router)
app.include_router(models.router)
app.include_router(pipelines.router)
app.include_router(projects.router)
app.include_router(sinks.router)
app.include_router(sources.router)
app.include_router(system.router)
app.include_router(training_configurations.router)
app.include_router(webrtc.router)

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
async def health_check(
    license_service: Annotated[LicenseService, Depends(get_license_service)],
) -> dict[str, str | bool]:
    """Health check endpoint"""
    try:
        license_accepted = license_service.is_accepted()
    except OSError:
        logger.warning("License acceptance check failed during health check", exc_info=True)
        return {"status": "degraded", "license_accepted": False}

    return {"status": "ok", "license_accepted": license_accepted}


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


@app.exception_handler(ResourceNotFoundError)
async def resource_not_found_exception_handler(request: Request, exc: ResourceNotFoundError) -> JSONResponse:  # noqa: ARG001
    """Catch resource not found errors and return 404 response"""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc)},
    )


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
