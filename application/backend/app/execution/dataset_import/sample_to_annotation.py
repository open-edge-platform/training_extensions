# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datumaro.experimental.categories import LabelCategories
from loguru import logger

from app.datumaro_converter import (
    ClassificationSample,
    DetectionSample,
    InstanceSegmentationSample,
    MultilabelClassificationSample,
)
from app.datumaro_converter.domain import LabelIndex
from app.models import DatasetItemAnnotation, FullImage, Label, LabelReference, Point, Polygon, Rectangle
from app.utils.typing import NDArrayFloat32, NDArrayInt

SampleType = ClassificationSample | MultilabelClassificationSample | DetectionSample | InstanceSegmentationSample


class DatumaroSampleToGetiAnnotationConverter:
    """
    Converts Datumaro sample labels to Geti dataset annotations.

    This class provides methods to transform Datumaro annotation objects
    (such as labels, bounding boxes, polygons, etc.) into the format expected
    by Geti datasets.

    Methods:
        convert_sample: Convert a complete Datumaro sample to Geti annotation.

    Example:
        >>> converter = DatumaroSampleToGetiAnnotationConverter(project_labels, label_categories)
        >>> geti_annotations = converter.convert_sample(datumaro_sample)
    """

    def __init__(
        self, project_labels: list[Label], label_categories: LabelCategories, label_mapping: dict[str, str] | None
    ) -> None:
        """Initialize the Datumaro to Geti converter."""
        self._label_categories = label_categories
        self._project_labels = LabelIndex(project_labels)
        self._label_mapping = label_mapping or {}

    def convert_sample(self, sample: SampleType) -> list[DatasetItemAnnotation] | None:
        """
        Convert all annotations from a Datumaro sample to Geti format.

        Args:
            sample: A Datumaro sample containing annotations.

        Returns:
            A list of Geti annotations.

        Raises:
            ValueError: If the sample format is invalid or unsupported.
        """
        match sample:
            case ClassificationSample(label=label, confidence=confidence):
                return self.__convert_classification_sample(label, confidence)
            case MultilabelClassificationSample(label=labels, confidence=confidences):
                return self.__convert_multilabel_sample(labels, confidences)
            case DetectionSample(label=labels, bboxes=bboxes, confidence=confidences):
                return self.__convert_detection_sample(labels, bboxes, confidences)
            case InstanceSegmentationSample(label=labels, polygons=polygons, confidence=confidences):
                return self.__convert_segmentation_sample(labels, polygons, confidences)

    def __convert_classification_sample(
        self, label: int | None, confidence: float | None
    ) -> list[DatasetItemAnnotation]:
        if label_refs := self.__convert_labels_to_refs(label):
            return [
                DatasetItemAnnotation(
                    shape=FullImage(),
                    labels=label_refs,
                    confidences=[confidence] if confidence else None,
                )
            ]
        return []

    def __convert_multilabel_sample(
        self, labels: NDArrayInt, confidences: NDArrayFloat32 | None
    ) -> list[DatasetItemAnnotation]:
        if label_refs := self.__convert_labels_to_refs(labels):
            return [
                DatasetItemAnnotation(
                    shape=FullImage(),
                    labels=label_refs,
                    confidences=confidences,
                )
            ]
        return []

    def __convert_detection_sample(
        self, labels: NDArrayInt, bboxes: NDArrayInt, confidences: NDArrayFloat32 | None
    ) -> list[DatasetItemAnnotation]:
        annotations = []
        if label_refs := self.__convert_labels_to_refs(labels):
            for idx, bbox in enumerate(bboxes):
                label_ref = label_refs[idx]
                x1, y1, x2, y2 = bbox
                annotations.append(
                    DatasetItemAnnotation(
                        labels=[label_ref],
                        shape=Rectangle(x=x1, y=y1, width=(x2 - x1), height=(y2 - y1)),
                        confidences=[confidences[idx]] if confidences is not None else None,
                    )
                )
        return annotations

    def __convert_segmentation_sample(
        self, labels: NDArrayInt, polygons: NDArrayFloat32, confidences: NDArrayFloat32 | None
    ) -> list[DatasetItemAnnotation]:
        annotations = []
        if label_refs := self.__convert_labels_to_refs(labels):
            for idx, polygon in enumerate(polygons):
                label_ref = label_refs[idx]
                annotations.append(
                    DatasetItemAnnotation(
                        labels=[label_ref],
                        shape=Polygon(points=[Point(x=x, y=y) for (x, y) in polygon]),
                        confidences=[confidences[idx]] if confidences is not None else None,
                    )
                )
        return annotations

    def __convert_labels_to_refs(self, label_idx: int | NDArrayInt | None) -> list[LabelReference]:
        if label_idx is None:
            return []
        label_refs = []
        if isinstance(label_idx, int):
            label_ref = self.__get_label_ref_by_dm_index(label_idx)
            if label_ref is not None:
                label_refs.append(label_ref)
        elif isinstance(label_idx, NDArrayInt):
            label_refs = []
            for idx in label_idx:
                label_ref = self.__get_label_ref_by_dm_index(idx)
                if label_ref is not None:
                    label_refs.append(label_ref)
        return label_refs

    def __get_label_ref_by_dm_index(self, dm_label_idx: int) -> LabelReference | None:
        """Get a Geti label reference by the index of a Datumaro label."""
        label_name = str(self._label_categories[dm_label_idx])
        mapped_name = self._label_mapping.get(label_name, label_name)
        label_id = self._project_labels.get_id_by_name(mapped_name)
        if label_id is None:
            logger.warning("Label {} not found in project labels", label_name)
            return None
        return LabelReference(id=label_id)
