# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Package-wide loguru logger.

Logging is disabled for ``getitrack`` by default (see
``getitrack/__init__.py``); the library stays silent until the application
calls ``logger.enable("getitrack")`` or a tracker runs with ``verbose``.
"""

from __future__ import annotations

from loguru import logger

LOGGER = logger


def enable_logging() -> None:
    """Lift the default suppression of getitrack log records."""
    logger.enable("getitrack")
