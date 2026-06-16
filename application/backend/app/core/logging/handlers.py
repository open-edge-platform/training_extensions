# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import inspect
import logging
from typing import TextIO

from loguru import logger


class InterceptHandler(logging.Handler):
    """
    This handler intercepts standard logging calls and forwards them to loguru
    while preserving the original caller information.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


class LoggerStdoutWriter:
    """File-like wrapper that forwards stdout/stderr writes to loguru."""

    def __init__(self, original_stream: TextIO, level: str = "INFO") -> None:
        self._original_stream = original_stream
        self._level = logger.level(level).name
        self._buffer: list[str] = []

    def write(self, msg: str) -> int:
        for char in msg:
            if char == "\r":
                self._buffer.clear()
                continue
            if char == "\n":
                self._emit_buffer()
                continue
            self._buffer.append(char)
        return len(msg)

    def flush(self) -> None:
        self._emit_buffer()

    def isatty(self) -> bool:
        return bool(getattr(self._original_stream, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._original_stream.fileno()

    @property
    def encoding(self) -> str:
        return getattr(self._original_stream, "encoding", "utf-8")

    def _emit_buffer(self) -> None:
        msg = "".join(self._buffer).strip()
        if msg:
            logger.log(self._level, msg)
        self._buffer.clear()
