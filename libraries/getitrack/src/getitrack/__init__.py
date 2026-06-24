# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""getitrack: Multi-object tracking toolkit."""

from loguru import logger as _logger

import getitrack.algorithms  # noqa: F401  -> registers the bundled algorithms on import

# Silent by default; the application opts in with ``logger.enable("getitrack")``.
_logger.disable("getitrack")

__version__ = "0.1.0"
