# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom StaticFiles handler that adds Cache-Control headers to static asset responses."""

from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from starlette.types import Scope


class CachedStaticFiles(StaticFiles):
    """StaticFiles subclass that sets Cache-Control headers based on file extension."""

    CACHE_RULES: list[tuple[tuple[str, ...], str]] = [
        # Fingerprinted JS/CSS bundles — safe to cache for one year
        ((".js", ".css"), "public, max-age=31536000, immutable"),
        # WebAssembly + OpenCV — large binary assets cached immutably for one year
        ((".wasm",), "public, max-age=31536000, immutable"),
        # OpenCV data files (e.g. haarcascades, trained models)
        ((".data", ".mem"), "public, max-age=31536000, immutable"),
        # Images
        ((".png", ".jpg", ".jpeg", ".webp", ".svg", ".ico"), "public, max-age=604800"),
        # Fonts — rarely change, safe to cache for 30 days
        ((".woff", ".woff2", ".ttf", ".otf"), "public, max-age=2592000, immutable"),
        # HTML — always revalidate so users get fresh asset references
        ((".html",), "no-cache"),
    ]

    async def get_response(self, path: str, scope: Scope) -> Response:
        """Get response with appropriate Cache-Control header based on file extension."""
        response = await super().get_response(path, scope)
        for extensions, cache_value in self.CACHE_RULES:
            if path.endswith(extensions):
                response.headers["Cache-Control"] = cache_value
                break
        else:
            # Fallback for anything not explicitly matched
            response.headers["Cache-Control"] = "public, max-age=3600"
        return response
