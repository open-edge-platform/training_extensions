# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence

import numpy as np
from datumaro.experimental.fields import ImageInfo
from loguru import logger

from app.datumaro_converter.samples import MultilabelClassificationSample
from app.datumaro_converter.utils import SubsetConverter, validate_confidence_consistency
from app.models import DatasetItem, DatasetItemAnnotation, Media

from .sample_factory import SampleFactory


class MultilabelClassificationSampleFactory(SampleFactory):
    """Knows how to create multilabel classification samples."""

    @property
    def sample_type(self) -> type[MultilabelClassificationSample]:
        return MultilabelClassificationSample

    def create_sample(
        self, dataset_item: DatasetItem, media: Media, image_path: str
    ) -> MultilabelClassificationSample | None:
        if dataset_item.annotation_data is None:
            return MultilabelClassificationSample(
                id=str(dataset_item.id),
                image=image_path,
                image_info=ImageInfo(width=media.width, height=media.height),
                subset=SubsetConverter.to_datumaro(dataset_item.subset),
                label=np.array([]),
                confidence=None,
                # user_reviewed=False,
            )

        label_indices = self._extract_label_indices(dataset_item)
        if label_indices is None:
            return None

        confidences = self._extract_confidences(dataset_item.annotation_data)

        return MultilabelClassificationSample(
            id=str(dataset_item.id),
            image=image_path,
            image_info=ImageInfo(width=media.width, height=media.height),
            label=np.array(label_indices),
            confidence=np.array(confidences) if confidences else None,
            subset=SubsetConverter.to_datumaro(dataset_item.subset),
            # user_reviewed=True,
        )

    def _extract_label_indices(self, dataset_item: DatasetItem) -> list[int] | None:
        """Extracts indices for all labels in the annotations."""
        label_ids = [label.id for ann in dataset_item.annotation_data for label in ann.labels]  # pyrefly: ignore

        indices = self._label_index.get_indices(label_ids)

        if indices is None:
            logger.error(f"Label not found for dataset item {dataset_item.id}")

        return indices

    @staticmethod
    def _extract_confidences(annotations: Sequence[DatasetItemAnnotation]) -> list[float]:
        """Extracts confidence scores if consistently present."""
        has_confidences = validate_confidence_consistency(annotations)

        if not has_confidences:
            return []

        confidences = []
        for ann in annotations:
            if ann.confidences:
                confidences.extend(ann.confidences)

        return confidences
