# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datumaro.experimental.fields import ImageInfo
from loguru import logger

from app.datumaro_converter.domain import ClassificationSample
from app.datumaro_converter.utils import SubsetConverter
from app.models import DatasetItem, Media

from .sample_factory import SampleFactory


class ClassificationSampleFactory(SampleFactory):
    """Knows how to create classification samples."""

    @property
    def sample_type(self) -> type[ClassificationSample]:
        return ClassificationSample

    def create_sample(self, dataset_item: DatasetItem, media: Media, image_path: str) -> ClassificationSample | None:
        if dataset_item.annotation_data is None:
            return ClassificationSample(
                id=str(dataset_item.id),
                image=image_path,
                image_info=ImageInfo(width=media.width, height=media.height),
                subset=SubsetConverter.to_datumaro(dataset_item.subset),
                label=None,
                confidence=None,
                user_reviewed=False,
            )

        if len(dataset_item.annotation_data) == 0:
            raise ValueError(
                f"Expected exactly one annotation for classification project, found empty list. "
                f"Project ID {dataset_item.project_id}, Dataset Item ID: {dataset_item.id}"
            )
        annotation = dataset_item.annotation_data[0]
        label_index = self._label_index.get_index(annotation.labels[0].id)

        if label_index is None:
            logger.error(f"Label not found for dataset item {dataset_item.id}")
            return None

        return ClassificationSample(
            id=str(dataset_item.id),
            image=image_path,
            image_info=ImageInfo(width=media.width, height=media.height),
            label=label_index,
            confidence=annotation.confidences[0] if annotation.confidences else None,
            subset=SubsetConverter.to_datumaro(dataset_item.subset),
            user_reviewed=True,
        )
