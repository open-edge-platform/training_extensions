# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Generator
from io import BytesIO
from pathlib import Path

from PIL.Image import Image
from starlette.responses import StreamingResponse


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


def write_bytes_to_response(
    bytes: BytesIO | Generator[bytes], filename: str, media_type: str | None = None, cache_control: str | None = None
) -> StreamingResponse:
    """
    Stream a binary content to FastAPI StreamingResponse.
    Additionally, method sets file name, MIME type and cache control headers if provided,

    Args:
        bytes: Binary content.
        filename: File name.
        media_type: Content MIME type, optional.
        cache_control: Cache control header, optional.

    Returns:
        FastAPI StreamingResponse.
    """
    headers = {"Content-Disposition": f"inline; filename={filename}"}
    if cache_control:
        headers["Cache-Control"] = cache_control
    return StreamingResponse(bytes, media_type=media_type, headers=headers)


def write_image_to_response(image: Image, filename: str, cache_control: str | None = None) -> StreamingResponse:
    """
    Stream a Pillow image as JPEG to FastAPI StreamingResponse.
    Additionally, method sets file name and cache control headers if provided.

    Args:
        image: Pillow image.
        filename: File name.
        cache_control: Cache control header, optional.

    Returns:
        FastAPI StreamingResponse.
    """
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return write_bytes_to_response(
        bytes=buffer, filename=filename, media_type="image/jpeg", cache_control=cache_control
    )


def write_file_to_response(path: Path, filename: str, cache_control: str | None = None) -> StreamingResponse:
    """
    Stream a file from file system to FastAPI StreamingResponse.
    Additionally, method sets file name and cache control headers if provided.

    Args:
        path: Path to the file.
        filename: File name.
        cache_control: Cache control header, optional.

    Returns:
        FastAPI StreamingResponse.
    """
    return write_bytes_to_response(bytes=file_iterator(filepath=path), filename=filename, cache_control=cache_control)
