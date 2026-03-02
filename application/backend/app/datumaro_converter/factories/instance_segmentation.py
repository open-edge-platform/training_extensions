# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import numpy as np
from datumaro.experimental.fields import ImageInfo
from loguru import logger
from numpy import ndarray

from app.datumaro_converter.domain import InstanceSegmentationSample
from app.datumaro_converter.utils import ShapeConverter, SubsetConverter, validate_confidence_consistency
from app.models import DatasetItem, DatasetItemAnnotation, DatasetItemSubset, Media, Polygon

from .sample_factory import SampleFactory


class InstanceSegmentationSampleFactory(SampleFactory):
    """Knows how to create instance segmentation samples."""

    @property
    def sample_type(self) -> type[InstanceSegmentationSample]:
        return InstanceSegmentationSample

    def create_sample(
        self, dataset_item: DatasetItem, media: Media, image_path: str
    ) -> InstanceSegmentationSample | None:
        if dataset_item.annotation_data is None:
            return InstanceSegmentationSample(
                id=str(dataset_item.id),
                image=image_path,
                image_info=ImageInfo(width=media.width, height=media.height),
                subset=SubsetConverter.to_datumaro(dataset_item.subset),
                polygons=np.array([]),
                label=np.array([]),
                confidence=None,
                user_reviewed=False,
            )

        polygons = self._extract_polygons(dataset_item.annotation_data)
        label_indices = self._extract_label_indices(dataset_item)

        if label_indices is None:
            return None
        # Filter out empty-labeled items in the training subset
        if len(label_indices) == 0 and dataset_item.subset == DatasetItemSubset.TRAINING:
            return None

        confidences = self._extract_confidences(dataset_item.annotation_data)

        return InstanceSegmentationSample(
            id=str(dataset_item.id),
            image=image_path,
            image_info=ImageInfo(width=media.width, height=media.height),
            polygons=polygons,
            label=np.array(label_indices),
            confidence=np.array(confidences) if confidences else None,
            subset=SubsetConverter.to_datumaro(dataset_item.subset),
            user_reviewed=True,
        )

    def _extract_polygons(self, annotations: Sequence[DatasetItemAnnotation]) -> ndarray:
        """Extracts polygon coordinates from annotations."""
        polygons = [
            ShapeConverter.polygon_to_points(ann.shape) for ann in annotations if isinstance(ann.shape, Polygon)
        ]
        # Each polygon is a float32 array with dimension (num_vert, 2); since the number of vertices varies
        # between polygons, we need a ragged array (dtype=object) to stack the polygons.
        # While building this array, we must also avoid unwanted type conversions by numpy, hence the use of np.empty
        # and explicit assignment - trying to wrap the polygons simply with np.array([...], dtype=object) would cause
        # the outer array dtype to become float32 instead of object if all polygons have the same number of vertices.
        polygons_np = np.empty(len(polygons), dtype=object)
        polygons_np[:] = [np.asarray(p, dtype=np.float32) for p in polygons]
        return polygons_np

    def _extract_label_indices(self, dataset_item: DatasetItem) -> list[int] | None:
        """Extracts label indices ensuring single label per annotation."""
        label_ids = [
            ann.labels[0].id
            for ann in dataset_item.annotation_data  # pyrefly: ignore
            if len(ann.labels) == 1
        ]

        indices = self._label_index.get_indices(label_ids)

        if indices is None:
            logger.warning(f"Label not found for dataset item {dataset_item.id}")

        return indices

    def _extract_confidences(self, annotations: Sequence[DatasetItemAnnotation]) -> list[float]:
        """Extracts confidence scores if consistently present."""
        has_confidences = validate_confidence_consistency(annotations)
        return [ann.confidences[0] for ann in annotations] if has_confidences else []  # pyrefly: ignore
