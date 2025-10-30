# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# export no_proxy="localhost, 127.0.0.1, ::1"
# Start with:
#  - uv run app/main.py
# or use docker and access UI and backend at geti-tune.localhost
#  - docker compose up

import importlib
import logging
import pkgutil
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api import routers
from app.lifecycle import lifespan
from app.settings import get_settings

settings = get_settings()

logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

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
    allow_origins=["*"],
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


def main() -> None:
    """Main application entry point"""
    logger.info(f"Starting {settings.app_name} in {settings.environment} mode")
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        # FIXME: reload mode currently does not work with multiple workers
        # reload=settings.debug and settings.environment == "dev",
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    main()
