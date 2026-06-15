# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Package-wide logger.

Import `LOGGER` instead of ``print`` or per-module loggers::

    from getitrack.logger import LOGGER
    LOGGER.info("...")

Libraries must not configure the root logger, so output stays invisible
until either the application configures logging or `enable_console_output`
runs (the tracker calls it automatically when ``verbose`` is enabled).
"""

from __future__ import annotations

import logging

LOGGER = logging.getLogger("getitrack")


def enable_console_output() -> None:
    """Make INFO-level getitrack logs visible when nothing configured logging.

    Attaches a plain stream handler to the ``getitrack`` logger, and only
    when the application has not configured logging itself. Applications
    that did configure logging keep full control over the output.
    """
    if logging.getLogger().handlers:
        return  # the application configured logging; respect it
    if LOGGER.handlers:
        return  # already enabled
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
