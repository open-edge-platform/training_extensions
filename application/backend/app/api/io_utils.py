# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Generator
from pathlib import Path


def file_iterator(filepath: Path, chunk_size: int = 1024 * 1024) -> Generator[bytes]:
    """
    Stream a file from disk in fixed-size binary chunks.

    This helper opens the file at `filepath` for reading in binary mode and yields its contents in chunks of
    `chunk_size` bytes. It is suitable for use with streaming HTTP responses where the whole file should not be loaded
    into memory at once.

    Args:
        filepath: Path to the file that will be streamed.
        chunk_size: Maximum number of bytes to read and yield per chunk.

    Yields:
        Consecutive chunks of the file content as `bytes` until EOF.
    """
    with filepath.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk
