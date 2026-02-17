# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence

import numpy as np
from datumaro.experimental.fields import ImageInfo
from loguru import logger

from app.datumaro_converter.domain import DetectionSample
from app.datumaro_converter.utils import ShapeConverter, SubsetConverter, validate_confidence_consistency
from app.models import DatasetItem, DatasetItemAnnotation, DatasetItemSubset, Media, Rectangle

from .sample_factory import SampleFactory


class DetectionSampleFactory(SampleFactory):
    """Knows how to create detection samples."""

    @property
    def sample_type(self) -> type[DetectionSample]:
        return DetectionSample

    def create_sample(self, dataset_item: DatasetItem, media: Media, image_path: str) -> DetectionSample | None:
        if dataset_item.annotation_data is None:
            return DetectionSample(
                id=str(dataset_item.id),
                image=image_path,
                image_info=ImageInfo(width=media.width, height=media.height),
                bboxes=np.array([]),
                label=np.array([]),
                confidence=None,
                subset=SubsetConverter.to_datumaro(dataset_item.subset),
                user_reviewed=False,
            )

        bboxes = self._extract_bboxes(dataset_item.annotation_data)
        label_indices = self._extract_label_indices(dataset_item)

        if label_indices is None:
            return None
        # Filter out empty-labeled items in the training subset
        if len(label_indices) == 0 and dataset_item.subset == DatasetItemSubset.TRAINING:
            return None

        confidences = self._extract_confidences(dataset_item.annotation_data)

        return DetectionSample(
            id=str(dataset_item.id),
            image=image_path,
            image_info=ImageInfo(width=media.width, height=media.height),
            bboxes=np.array(bboxes),
            label=np.array(label_indices),
            confidence=np.array(confidences) if confidences else None,
            subset=SubsetConverter.to_datumaro(dataset_item.subset),
            user_reviewed=True,
        )

    @staticmethod
    def _extract_bboxes(annotations: Sequence[DatasetItemAnnotation]) -> list[list[int]]:
        return [ShapeConverter.rectangle_to_bbox(ann.shape) for ann in annotations if isinstance(ann.shape, Rectangle)]

    def _extract_label_indices(self, dataset_item: DatasetItem) -> list[int] | None:
        label_ids = [
            ann.labels[0].id
            for ann in dataset_item.annotation_data  # pyrefly: ignore
            if len(ann.labels) == 1
        ]
        indices = self._label_index.get_indices(label_ids)

        if indices is None:
            logger.warning(f"Label not found for dataset item {dataset_item.id}")

        return indices

    @staticmethod
    def _extract_confidences(annotations: Sequence[DatasetItemAnnotation]) -> list[float]:
        has_confidences = validate_confidence_consistency(annotations)
        return [ann.confidences[0] for ann in annotations] if has_confidences else []  # pyrefly: ignore
