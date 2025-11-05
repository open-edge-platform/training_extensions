# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Generator
from contextlib import contextmanager

from loguru import logger

from .config import LogConfig


@contextmanager
def logging_ctx(config: LogConfig) -> Generator[str]:
    """Create a temporary logging context with an additional file sink.

    Adds a context-specific log file sink that captures all logs emitted within
    the context. The sink is automatically removed on exit, but the log file
    persists. Logs continue to go to all other configured sinks (stdout, main
    log file, etc.).

    Useful for capturing logs from specific operations (e.g., training jobs) into separate files while maintaining
    application-wide logging.

    Args:
        config: LogConfig instance specifying the log file path, rotation,
                retention, and other sink parameters.

    Yields:
        str: Full path to the created log file.

    Raises:
        RuntimeError: If the log sink cannot be added (e.g., due to permission
                      issues or invalid configuration).

    Example:
        >>> log_config = LogConfig(
        ...     log_folder="logs/jobs",
        ...     log_file="train-8f3e22f2.log"
        ... )
        >>> with logging_ctx(log_config) as logging_path:
        ...     logger.info("Training started")  # Logged to both main and job-specific file
        ...     # ... training code ...
        >>> # Sink removed, but logs/jobs/train-8f3e22f2.log persists
    """
    log_path = os.path.join(config.log_folder, config.log_file)

    try:
        sink_id = logger.add(
            log_path,
            rotation=config.rotation,
            retention=config.retention,
            level=config.level,
            serialize=config.serialize,
            enqueue=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to add log sink for {log_path}: {e}") from e

    try:
        logger.debug("Started logging to {}", log_path)
        yield log_path
    finally:
        logger.debug("Stopped logging to {}", log_path)
        logger.remove(sink_id)
