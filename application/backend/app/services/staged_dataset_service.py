# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable
from pathlib import Path
from uuid import UUID, uuid4

from app.models import DatasetFormat, StagedDataset


def _infer_format_from_filename(filename: str) -> DatasetFormat:
    """
    Infer the dataset format from an archive file name.

    The file name is matched case-insensitively against the string values of `DatasetFormat`. If any format value
    occurs as a substring of the lowercased file name, that format is returned;
    otherwise `DatasetFormat.UNKNOWN` is used.

    Args:
        filename: Name of the archive file (with or without extension).

    Returns:
        The inferred `DatasetFormat` based on the file name.
    """
    lower_name = filename.lower()
    for fmt in (f.value for f in DatasetFormat):
        if fmt in lower_name:
            return DatasetFormat(fmt)
    return DatasetFormat.UNKNOWN


class StagedDatasetService:
    def __init__(self, staged_datasets_dir: Path) -> None:
        self._staged_datasets_dir = staged_datasets_dir

    async def upload(self, filename: str, chunk_reader: Callable[[], Awaitable[bytes]]) -> StagedDataset:
        """
        Store an uploaded dataset archive into a new staged dataset directory.

        A new UUID is generated, a subdirectory with that UUID is created under the configured staging root,
        and the incoming byte stream is written to a file with the given filename.

        Args:
            filename: Target filename of the uploaded archive within the staged dataset directory.
            chunk_reader: Async callable that returns the next chunk of bytes from the upload stream.
                Must return an empty `bytes` object to signal end of stream.

        Returns:
            A `StagedDataset` object containing the dataset identifier, the total number of bytes written,
            the inferred dataset format based on the filename, and a `compressed` flag.
        """
        dataset_id = uuid4()
        target_dir = self._staged_datasets_dir / str(dataset_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / filename

        size = 0
        with target_path.open("wb") as out_f:
            while True:
                chunk = await chunk_reader()
                if not chunk:
                    break
                size += len(chunk)
                out_f.write(chunk)

        return StagedDataset(
            id=dataset_id,
            size=size,
            format=_infer_format_from_filename(filename),
            compressed=True,
        )

    def list_all(self) -> list[StagedDataset]:
        """
        List all staged dataset archives in the staging directory.

        Each staged dataset is expected to reside in a subdirectory whose name is a UUID. For every such directory,
        the first regular file found is treated as the archive.

        Returns:
            A list of `StagedDataset` objects, each containing the dataset identifier, the inferred dataset format,
            the size of the archive file in bytes, and a `compressed` flag indicating whether the archive is
            detected as compressed (currently `True` only for `.zip` files).
        """
        staged_datasets = []

        for item in self._staged_datasets_dir.iterdir():
            if not item.is_dir():
                continue

            dataset_id = UUID(item.name)
            files = [p for p in item.iterdir() if p.is_file()]
            if not files:
                continue

            archive_path = files[0]
            size = archive_path.stat().st_size
            compressed = archive_path.is_file() and archive_path.suffix == ".zip"
            dataset_format = _infer_format_from_filename(archive_path.name)
            staged_datasets.append(
                StagedDataset(
                    id=dataset_id,
                    format=dataset_format,
                    compressed=compressed,
                    size=size,
                )
            )
        return staged_datasets
