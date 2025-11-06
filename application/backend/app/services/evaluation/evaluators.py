# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABCMeta, abstractmethod
from enum import StrEnum

import numpy as np
from datumaro.experimental import Dataset
from faster_coco_eval import COCO, COCOeval_faster
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

from app.services.datumaro_converter import DetectionSample


def datumaro_dataset_to_coco(dataset: Dataset) -> dict:
    """
    Convert Datumaro Dataset to COCO format.

    Supports detection (DetectionSample) and instance segmentation (InstanceSegmentationSample) datasets.

    Args:
        dataset (Dataset): Datumaro Dataset to convert.
    Returns:
        dict: COCO formatted dataset.
    """
    coco_dataset_dict: dict[str, list] = {"images": [], "annotations": [], "categories": []}

    # Add categories
    for label_idx, label in enumerate(dataset.schema.attributes["label"].categories.labels):
        coco_dataset_dict["categories"].append(
            {
                "id": label_idx,
                "name": label,
            }
        )

    annotation_id = 1  # COCOeval ignores annotation ID 0
    for image_id, sample in enumerate(dataset):
        # Add image entry
        coco_dataset_dict["images"].append(
            {
                "id": image_id,
                "file_name": sample.image,
                "width": sample.image_info.width,
                "height": sample.image_info.height,
            }
        )

        # Detection
        if hasattr(sample, "bboxes") and sample.bboxes is not None:
            confidences = sample.confidence if sample.confidence is not None else [None] * len(sample.bboxes)
            for bbox, label_idx, confidence in zip(sample.bboxes, sample.label, confidences, strict=True):
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(label_idx),
                    "bbox": [float(x1), float(y1), float(width), float(height)],
                }
                if confidence is not None:
                    annotation["score"] = float(confidence)
                coco_dataset_dict["annotations"].append(annotation)
                annotation_id += 1

        # Instance Segmentation
        if hasattr(sample, "polygons") and sample.polygons is not None:
            confidences = sample.confidence if sample.confidence is not None else [None] * len(sample.polygons)
            for polygon, label_idx, confidence in zip(sample.polygons, sample.label, confidences, strict=True):
                flattened_polygon = [coord for point in polygon for coord in point]
                x_coords = [point[0] for point in polygon]
                y_coords = [point[1] for point in polygon]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                width = x_max - x_min
                height = y_max - y_min
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(label_idx),
                    "segmentation": [flattened_polygon],
                    "bbox": [float(x_min), float(y_min), float(width), float(height)],
                }
                if confidence is not None:
                    annotation["score"] = float(confidence)
                coco_dataset_dict["annotations"].append(annotation)
                annotation_id += 1

    return coco_dataset_dict


class AveragingMethod(StrEnum):
    MICRO = "micro"
    MACRO = "macro"
    WEIGHTED = "weighted"
    SAMPLES = "samples"


class EvaluatorBase(metaclass=ABCMeta):
    """Base class for all evaluators."""

    def __init__(self, predictions_dataset: Dataset, ground_truth_dataset: Dataset):
        self.predictions_dataset = predictions_dataset
        self.ground_truth_dataset = ground_truth_dataset


class EvaluatorWithLabelArrays(EvaluatorBase):
    """Base evaluator for tasks that use label arrays."""

    def __init__(self, predictions_dataset: Dataset, ground_truth_dataset: Dataset):
        super().__init__(predictions_dataset=predictions_dataset, ground_truth_dataset=ground_truth_dataset)
        self.__pred_labels: NDArray[np.int_] | None = None
        self.__gt_labels: NDArray[np.int_] | None = None

    @abstractmethod
    def _build_label_arrays(self) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        """Set up the prediction and ground truth label arrays."""

    @property
    def _pred_labels(self) -> NDArray[np.int_]:
        if self.__pred_labels is None:
            self.__gt_labels, self.__pred_labels = self._build_label_arrays()
        return self.__pred_labels

    @property
    def _gt_labels(self) -> NDArray[np.int_]:
        if self.__gt_labels is None:
            self.__gt_labels, self.__pred_labels = self._build_label_arrays()
        return self.__gt_labels


class AccuracyEvaluator(EvaluatorWithLabelArrays):
    """Evaluator for accuracy, precision, recall, and F1 metrics."""

    def __init__(self, predictions_dataset: Dataset, ground_truth_dataset: Dataset):
        super().__init__(predictions_dataset=predictions_dataset, ground_truth_dataset=ground_truth_dataset)

    def precision(self, averaging_method: AveragingMethod = AveragingMethod.MACRO) -> float:
        return precision_score(y_true=self._gt_labels, y_pred=self._pred_labels, average=averaging_method.value)

    def recall(self, averaging_method: AveragingMethod = AveragingMethod.MACRO) -> float:
        return recall_score(y_true=self._gt_labels, y_pred=self._pred_labels, average=averaging_method.value)

    def accuracy(self) -> float:
        return accuracy_score(y_true=self._gt_labels, y_pred=self._pred_labels)

    def f1_score(self, averaging_method: AveragingMethod = AveragingMethod.MACRO) -> float:
        return f1_score(y_true=self._gt_labels, y_pred=self._pred_labels, average=averaging_method.value)


class ConfusionMatrixEvaluator(EvaluatorWithLabelArrays):
    """Evaluator for confusion matrix computation."""

    def __init__(self, predictions_dataset: Dataset, ground_truth_dataset: Dataset):
        super().__init__(predictions_dataset=predictions_dataset, ground_truth_dataset=ground_truth_dataset)

    def confusion_matrix(self) -> np.ndarray:
        """Compute the confusion matrix"""
        return confusion_matrix(y_true=self._gt_labels, y_pred=self._pred_labels)


class MeanAveragePrecisionEvaluator(EvaluatorBase):
    """Evaluator for mean average precision (mAP) metrics."""

    def __init__(self, predictions_dataset: Dataset, ground_truth_dataset: Dataset):
        super().__init__(predictions_dataset=predictions_dataset, ground_truth_dataset=ground_truth_dataset)
        self.__gt_coco_dict: dict | None = None
        self.__pred_coco_dict: dict | None = None

    @property
    def _gt_coco_dict(self) -> dict:
        if self.__gt_coco_dict is None:
            self.__gt_coco_dict = datumaro_dataset_to_coco(self.ground_truth_dataset)
        return self.__gt_coco_dict

    @property
    def _pred_coco_dict(self) -> dict:
        if self.__pred_coco_dict is None:
            self.__pred_coco_dict = datumaro_dataset_to_coco(self.predictions_dataset)
        return self.__pred_coco_dict

    def mean_average_precision(self) -> dict:
        gt_coco = COCO(self._gt_coco_dict)
        pred_coco = gt_coco.loadRes(self._pred_coco_dict["annotations"])
        coco_evaluator = COCOeval_faster(
            cocoGt=gt_coco,
            cocoDt=pred_coco,
            iouType="bbox" if self.predictions_dataset.dtype is DetectionSample else "segm",
        )
        coco_evaluator.run()
        return coco_evaluator.stats_as_dict


class MultiClassClassificationEvaluator(AccuracyEvaluator, ConfusionMatrixEvaluator):
    """Evaluator for multi-class classification tasks."""

    def __init__(self, predictions_dataset: Dataset, ground_truth_dataset: Dataset):
        if (
            predictions_dataset.schema.attributes["label"].annotation.multi_label
            or ground_truth_dataset.schema.attributes["label"].annotation.multi_label
        ):
            raise ValueError(f"{self.__class__.__name__} should not be used for multi-label classification datasets")

        AccuracyEvaluator.__init__(
            self, predictions_dataset=predictions_dataset, ground_truth_dataset=ground_truth_dataset
        )
        ConfusionMatrixEvaluator.__init__(
            self, predictions_dataset=predictions_dataset, ground_truth_dataset=ground_truth_dataset
        )

    def _build_label_arrays(self) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        pred_labels = np.array([sample.label for sample in self.predictions_dataset], dtype=int)
        gt_labels = np.array([sample.label for sample in self.ground_truth_dataset], dtype=int)
        return gt_labels, pred_labels


class MultiLabelClassificationEvaluator(AccuracyEvaluator):
    """Evaluator for multi-label classification tasks."""

    def __init__(self, predictions_dataset: Dataset, ground_truth_dataset: Dataset):
        if not (
            predictions_dataset.schema.attributes["label"].annotation.multi_label
            and ground_truth_dataset.schema.attributes["label"].annotation.multi_label
        ):
            raise ValueError(f"{self.__class__.__name__} should only be used for multi-label classification datasets")

        AccuracyEvaluator.__init__(
            self, predictions_dataset=predictions_dataset, ground_truth_dataset=ground_truth_dataset
        )

    def _build_label_arrays(self) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        mlb = MultiLabelBinarizer()
        gt_labels_list = [s.label for s in self.ground_truth_dataset]
        pred_labels_list = [s.label for s in self.predictions_dataset]
        gt_labels = mlb.fit_transform(gt_labels_list)
        pred_labels = mlb.transform(pred_labels_list)
        return gt_labels, pred_labels


class DetectionEvaluator(MeanAveragePrecisionEvaluator):
    """Evaluator for object detection tasks."""


class InstanceSegmentationEvaluator(MeanAveragePrecisionEvaluator):
    """Evaluator for instance segmentation tasks."""


Evaluator = (
    MultiClassClassificationEvaluator
    | MultiLabelClassificationEvaluator
    | DetectionEvaluator
    | InstanceSegmentationEvaluator
)
