# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""getitrack: Multi-object tracking toolkit."""

from loguru import logger

# Stay silent until the application opts in via `logger.enable("getitrack")`
# or a tracker's `verbose` flag.
logger.disable("getitrack")

__version__ = "0.1.0"
