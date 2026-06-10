# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from app.models import AnnotationType, DatasetFormat, StagedDataset
from app.models.dataset import DatasetMetadata

if TYPE_CHECKING:
    from datumaro.experimental import Dataset

    from app.datumaro_converter.domain.samples.import_export import BaseImportExportSample

_ANNOTATION_SHAPE_ATTRS: list[tuple[str, AnnotationType]] = [
    ("bboxes", AnnotationType.BOUNDING_BOX),
    ("polygons", AnnotationType.POLYGON),
]
# labels should be checked after bboxes and polygons, as they can be present in all samples.
_ANNOTATION_LABEL_ATTRS: list[tuple[str, AnnotationType]] = [
    ("labels", AnnotationType.LABEL),
    ("label", AnnotationType.LABEL),
]


def _count_annotations(sample: "BaseImportExportSample") -> tuple[AnnotationType, int]:
    if (
        hasattr(sample, "annotation_type")
        and callable(getattr(sample, "annotation_type", None))
        and hasattr(sample, "annotations")
    ):
        return sample.annotation_type(), sample.annotations

    # collect non-empty values from the sample for all annotation attributes and their corresponding types
    ann_type_with_value: list[tuple[AnnotationType, Any]] = []
    for attr, ann_type in _ANNOTATION_SHAPE_ATTRS:
        if (value := getattr(sample, attr, None)) is not None:
            ann_type_with_value.append((ann_type, value))
    if not ann_type_with_value:
        for attr, ann_type in _ANNOTATION_LABEL_ATTRS:
            if (value := getattr(sample, attr, None)) is not None:
                ann_type_with_value.append((ann_type, value))
    if len(ann_type_with_value) > 1:
        value = ann_type_with_value[0][1]
        return AnnotationType.UNKNOWN, 1 if isinstance(value, int) else len(value)
    if len(ann_type_with_value) == 1:
        value = ann_type_with_value[0][1]
        return ann_type_with_value[0][0], 1 if isinstance(value, int) else len(value)

    return AnnotationType.UNKNOWN, 0


@dataclass
class _Counts:
    annotation_type: AnnotationType = AnnotationType.UNKNOWN
    num_annotations: int = 0
    num_images: int = 0
    num_frames: int = 0
    num_annotated_images: int = 0
    num_annotated_frames: int = 0
    video_paths: set[str] = field(default_factory=set)


def _get_dataset_metadata(dataset: "Dataset") -> DatasetMetadata:
    from datumaro.experimental import LazyImage, LazyVideoFrame

    labels = []
    label_attr_name = "label" if "label" in dataset.schema.attributes else "labels"
    label_attr = dataset.schema.attributes[label_attr_name]
    if label_attr and label_attr.categories and hasattr(label_attr.categories, "labels"):
        labels = label_attr.categories.labels

    counts = _Counts()
    for item in dataset:
        ann_type, annotation_count = _count_annotations(item)
        if ann_type != AnnotationType.UNKNOWN:
            counts.annotation_type = ann_type
        target_media_attrs = ("media", "image", "image_path")
        values = (getattr(item, media_attr, None) for media_attr in target_media_attrs)
        media = next((v for v in values if v is not None), None)
        match media:
            case LazyImage() | str():
                counts.num_images += 1
                counts.num_annotations += annotation_count
                counts.num_annotated_images += annotation_count > 0
            case LazyVideoFrame():
                video_path = str(media.video_path)
                if video_path not in counts.video_paths:
                    counts.num_frames += media.video_info.total_frames
                counts.video_paths.add(video_path)
                counts.num_annotations += annotation_count
                counts.num_annotated_frames += annotation_count > 0
            case _:
                raise ValueError(f"Unsupported media type: {type(media)}")

    return DatasetMetadata(
        num_images=counts.num_images,
        num_frames=counts.num_frames,
        annotation_type=counts.annotation_type,
        num_annotations=counts.num_annotations,
        labels=sorted(labels),
        num_videos=len(counts.video_paths),
        num_annotated_frames=counts.num_annotated_frames,
        num_annotated_images=counts.num_annotated_images,
    )


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
            compressed=True,
            filename=str(target_path),
            format=DatasetFormat.UNKNOWN,
            id=dataset_id,
            size=size,
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

            try:
                dataset_id = UUID(item.name)
            except ValueError:
                continue
            files = [p for p in item.iterdir() if p.is_dir() or (p.is_file() and p.suffix.lower() == ".zip")]
            if not files:
                continue

            dataset_path = files[0]
            staged_datasets.append(self._get_staged_dataset_from_path(dataset_id, dataset_path))
        return staged_datasets

    def find_by_id(self, dataset_id: UUID) -> StagedDataset | None:
        """
        Finds a single staged dataset by its identifier.

        The dataset is expected to reside in a subdirectory named with the given UUID. The first regular file found
        in that directory is treated as the archive and mapped to a `StagedDataset` instance.

        Args:
            dataset_id: Identifier of the staged dataset to retrieve.

        Returns:
            A `StagedDataset` object if the dataset exists and contains at least one file; otherwise `None`.
        """
        dataset_dir = self._staged_datasets_dir / str(dataset_id)
        if not dataset_dir.is_dir():
            return None

        files = [p for p in dataset_dir.iterdir() if p.is_dir() or (p.is_file() and p.suffix.lower() == ".zip")]
        if not files:
            return None

        dataset_path = files[0]
        return self._get_staged_dataset_from_path(dataset_id, dataset_path)

    def delete_by_id(self, dataset_id: UUID) -> bool:
        """
        Delete a staged dataset directory and its contents.

        Returns `True` if the directory existed and was removed, otherwise `False`.
        """
        dataset_dir = self._staged_datasets_dir / str(dataset_id)
        if not dataset_dir.is_dir():
            return False

        shutil.rmtree(dataset_dir)
        return True

    @staticmethod
    def _calculate_path_size(path: Path) -> int:
        """Calculate total size of a file or directory in bytes."""
        if path.is_file():
            return path.stat().st_size
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    def _get_staged_dataset_from_path(self, dataset_id: UUID, dataset_path: Path) -> StagedDataset:
        from datumaro.experimental import import_dataset

        size = self._calculate_path_size(dataset_path)
        compressed = dataset_path.is_file() and dataset_path.suffix == ".zip"
        dataset_format = _infer_format_from_filename(dataset_path.name) if compressed else DatasetFormat.GETI

        metadata = None
        if not compressed and dataset_format == DatasetFormat.GETI:
            metadata = _get_dataset_metadata(import_dataset(dataset_path))

        return StagedDataset(
            compressed=compressed,
            filename=str(dataset_path),
            format=dataset_format,
            id=dataset_id,
            size=size,
            metadata=metadata,
        )
