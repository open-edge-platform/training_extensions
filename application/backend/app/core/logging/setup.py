# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Logging configuration and utilities for the application.

Provides centralized logging setup using loguru with:
- Console (stdout) and file-based logging sinks
- Configurable log levels, rotation (default: 10MB), and retention (default: 10 days)
- JSON serialization support for structured logging
- Thread-safe async logging with multiprocessing support
- Uvicorn log interception for unified application logging
"""

import logging
import multiprocessing
import os
import sys

from loguru import logger

from .config import LogConfig
from .handlers import InterceptHandler

context = multiprocessing.get_context("spawn")


def setup_logging(config: LogConfig | None = None) -> None:
    """Configure loguru logging with optional custom settings.

    Sets up loguru with stdout and file-based logging sinks. By default, creates
    a single log file with rotation and retention policies. Can be customized via
    LogConfig to specify different log levels, rotation sizes, and output locations.

    Args:
        config: Optional LogConfig instance. If None, uses default configuration
                with INFO level, 10MB rotation, and 10-day retention.

    Note:
        - Must be called in each child process separately, as loguru sinks don't
          transfer across process boundaries
        - BaseProcessWorker calls this automatically for worker processes
        - Call once at main process startup for application-level logging

    Example:
        >>> setup_logging()  # Uses defaults
        >>> custom_config = LogConfig(rotation="50 MB", level="DEBUG")
        >>> setup_logging(custom_config)
    """
    if config is None:
        config = LogConfig()

    logger.remove()

    logger.add(sys.stdout, level=config.level, colorize=True, enqueue=True, context=context)

    log_path = os.path.join(config.log_folder, config.log_file)
    try:
        logger.add(
            log_path,
            rotation=config.rotation,
            retention=config.retention,
            level=config.level,
            serialize=config.serialize,
            enqueue=True,
            context=context,
        )
    except Exception:
        logger.exception("Failed to add log sink for {}", log_path)


def setup_uvicorn_logging(log_level: str) -> None:
    """Configure uvicorn logging to be handled by loguru.

    Intercepts all uvicorn log messages (from uvicorn.error, uvicorn.access, etc.)
    and redirects them to loguru for unified logging output. This ensures uvicorn
    logs follow the same format and routing as application logs.

    The function:
    1. Attaches InterceptHandler to the main uvicorn logger
    2. Sets propagate=False on uvicorn logger to prevent duplicate logs to root
    3. Clears handlers from child loggers (uvicorn.access, uvicorn.error)
    4. Enables propagation on child loggers so they forward to parent uvicorn logger

    Note: This should be called during application startup, typically before
    starting the uvicorn server.

    Example:
        >>> setup_uvicorn_logging("INFO")
        # All uvicorn logs now flow through loguru
    """
    # Setup uvicorn logs to be handled by loguru
    # Configure the main uvicorn logger with InterceptHandler
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.handlers = [InterceptHandler()]
    uvicorn_logger.setLevel(log_level)
    uvicorn_logger.propagate = False  # Don't propagate to root to avoid duplicate logs
    # Clear handlers from child loggers and let them propagate to parent uvicorn logger
    for logger_name in ("uvicorn.access", "uvicorn.error"):
        child_logger = logging.getLogger(logger_name)
        child_logger.handlers.clear()
        child_logger.propagate = True  # Propagate to parent uvicorn logger
