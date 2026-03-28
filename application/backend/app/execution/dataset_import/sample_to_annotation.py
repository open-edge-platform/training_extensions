# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from datumaro.experimental.categories import LabelCategories

from app.datumaro_converter import (
    DetectionImportExportSample,
    InstanceSegmentationImportExportSample,
    MulticlassClassificationImportExportSample,
    MultilabelClassificationImportExportSample,
)
from app.datumaro_converter.domain import LabelIndex
from app.models import DatasetItemAnnotation, FullImage, Label, LabelReference, Point, Polygon, Rectangle
from app.utils.typing import NDArrayFloat32, NDArrayInt

SampleType = (
    MulticlassClassificationImportExportSample
    | DetectionImportExportSample
    | InstanceSegmentationImportExportSample
    | MultilabelClassificationImportExportSample
)


class DatumaroSampleToGetiAnnotationConverter:
    """
    Converts Datumaro sample labels to Geti dataset annotations.

    This class provides methods to transform Datumaro annotation objects
    (such as labels, bounding boxes, polygons, etc.) into the format expected
    by Geti datasets.

    Label Mapping Behavior:
        The converter supports label mapping during import to existing projects:

        - **Mapped labels**: Labels can be mapped to existing project labels.
          These annotations are converted normally with the mapped label name.

        - **Unmapped labels**: Labels that are not present in the mapping dictionary
          and in project labels will raise a validation error during import.

        - **Explicitly mapped to None**: Labels explicitly mapped to None (e.g.,
          `{"label_name": None}`) are filtered out during import. Annotations using
          these labels are stripped, which may result in some images becoming
          unannotated if all their annotations are removed.

    Methods:
        convert_sample: Convert a complete Datumaro sample to Geti annotation.

    Example:
        >>> converter = DatumaroSampleToGetiAnnotationConverter(project_labels, label_categories)
        >>> geti_annotations = converter.convert_sample(datumaro_sample)
    """

    def __init__(
        self,
        project_labels: list[Label],
        label_categories: LabelCategories,
        label_mapping: dict[str, str | None] | None,
    ) -> None:
        """Initialize the Datumaro to Geti converter."""
        self._label_categories = label_categories
        self._project_labels = LabelIndex(project_labels)
        self._label_mapping = label_mapping or {}
        self.__validate_labels(self._label_categories.labels, self._project_labels.label_names, self._label_mapping)

    @staticmethod
    def __validate_labels(imported: tuple[str, ...], existing: tuple[str, ...], mapping: dict[str, str | None]) -> None:
        existing_set = set(existing)

        # Find imported labels that are neither mapped nor present in the project
        # Example: imported=("car", "person"), existing=("car",), mapping={}
        # Result: bad=["person"] (not in project and not mapped)
        bad = [x for x in imported if x not in mapping and x not in existing_set]

        # Find mapping entries where the target label doesn't exist in the project
        # Example: mapping={"vehicle": "car", "human": "invalid_label"}, existing=("car",)
        # Result: bad_targets=[("human", "invalid_label")] (target doesn't exist)
        # Note: None values are allowed for filtering (e.g., {"background": None})
        bad_targets = [(k, v) for k, v in mapping.items() if v is not None and v not in existing_set]

        if bad or bad_targets:
            msg = "Label validation error during import:\n"
            if bad:
                msg += f" - Unmapped labels not in project: {bad}\n"
            if bad_targets:
                msg += f" - Mapped labels with invalid targets: {bad_targets}\n"
            raise ValueError(msg)

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
            case MulticlassClassificationImportExportSample(label=label, confidence=confidence):
                return self.__convert_multiclass_sample(label, confidence)
            case MultilabelClassificationImportExportSample(label=labels, confidence=confidences):
                return self.__convert_multilabel_sample(labels, confidences)
            case DetectionImportExportSample(label=labels, bboxes=bboxes, confidence=confidences):
                return self.__convert_detection_sample(labels, bboxes, confidences)
            case InstanceSegmentationImportExportSample(label=labels, polygons=polygons, confidence=confidences):
                return self.__convert_segmentation_sample(labels, polygons, confidences)
            case _:
                raise ValueError(f"Unsupported sample type: {type(sample)}")

    def __convert_multiclass_sample(
        self, label: int | None, confidence: float | None
    ) -> list[DatasetItemAnnotation] | None:
        if (label_refs := self.__convert_labels_to_refs(label)) and label_refs[0] is not None:
            return [
                DatasetItemAnnotation(
                    shape=FullImage(),
                    labels=label_refs,
                    confidences=[confidence] if confidence is not None else None,
                )
            ]
        return None

    def __convert_multilabel_sample(
        self, labels: NDArrayInt | None, confidences: NDArrayFloat32 | None
    ) -> list[DatasetItemAnnotation] | None:
        if label_refs := self.__convert_labels_to_refs(labels):
            return [
                DatasetItemAnnotation(
                    shape=FullImage(),
                    labels=label_refs,
                    confidences=confidences,
                )
            ]
        return None

    def __convert_detection_sample(
        self, labels: NDArrayInt | None, bboxes: NDArrayInt | None, confidences: NDArrayFloat32 | None
    ) -> list[DatasetItemAnnotation]:
        annotations = []
        if bboxes is not None and (label_refs := self.__convert_labels_to_refs(labels)):
            for idx, (x1, y1, x2, y2) in enumerate(bboxes):
                if (label_ref := label_refs[idx]) is not None:
                    annotations.append(
                        DatasetItemAnnotation(
                            labels=[label_ref],
                            shape=Rectangle(x=x1, y=y1, width=(x2 - x1), height=(y2 - y1)),
                            confidences=[confidences[idx]] if confidences is not None else None,
                        )
                    )
        return annotations

    def __convert_segmentation_sample(
        self, labels: NDArrayInt | None, polygons: NDArrayFloat32 | None, confidences: NDArrayFloat32 | None
    ) -> list[DatasetItemAnnotation]:
        annotations = []
        if polygons is not None and (label_refs := self.__convert_labels_to_refs(labels)):
            for idx, polygon in enumerate(polygons):
                if (label_ref := label_refs[idx]) is not None:
                    annotations.append(
                        DatasetItemAnnotation(
                            labels=[label_ref],
                            shape=Polygon(points=[Point(x=x, y=y) for (x, y) in polygon]),
                            confidences=[confidences[idx]] if confidences is not None else None,
                        )
                    )
        return annotations

    def __convert_labels_to_refs(self, label_idx: int | NDArrayInt | None) -> list[LabelReference | None]:
        if label_idx is None:
            return []
        if isinstance(label_idx, int):
            return [self.__get_label_ref_by_dm_index(label_idx)]
        if isinstance(label_idx, np.ndarray):
            if label_idx.ndim != 1:
                raise ValueError(f"Expected 1D array for label indices, got {label_idx.ndim}D")
            return [self.__get_label_ref_by_dm_index(idx) for idx in label_idx]
        raise ValueError(f"Unsupported label index type: {type(label_idx)}")

    def __get_label_ref_by_dm_index(self, dm_label_idx: int) -> LabelReference | None:
        """Get a Geti label reference by the index of a Datumaro label."""
        label_name = str(self._label_categories[dm_label_idx])
        mapped_name = self._label_mapping.get(label_name, label_name)
        if mapped_name is not None:
            label_id = self._project_labels.get_id_by_name(mapped_name)
            if label_id is not None:
                return LabelReference(id=label_id)
        return None
