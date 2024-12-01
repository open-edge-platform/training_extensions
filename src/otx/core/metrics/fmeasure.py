# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX custom f1 metrices."""

from __future__ import annotations

import inspect
import logging
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import tv_tensors
from torchvision.ops import box_iou

from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.detection import DetDataEntity, DetPredEntity
from otx.core.types.label import LabelInfo

logger = logging.getLogger()
ALL_CLASSES_NAME = "All Classes"


def get_n_false_negatives(iou_matrix: Tensor, iou_threshold: float) -> Tensor:
    """Get the number of false negatives inside the IoU matrix for a given threshold.

    The first loop accounts for all the ground truth boxes which do not have a high enough iou with any predicted
    box (they go undetected)
    The second loop accounts for the much rarer case where two ground truth boxes are detected by the same predicted
    box. The principle is that each ground truth box requires a unique prediction box

    Args:
        iou_matrix (torch.Tensor): IoU matrix of shape [ground_truth_boxes, predicted_boxes]
        iou_threshold (float): IoU threshold to use for the false negatives.

    Returns:
        Tensor: Number of false negatives
    """
    # First loop
    n_false_negatives = 0
    values = torch.max(iou_matrix, 1)[0] < iou_threshold - 1e-6  # 1e-6 is to avoid numerical instability
    n_false_negatives += sum(values)

    # Second loop
    matrix = torch.sum(iou_threshold < iou_matrix.T, 1)
    n_false_negatives += sum(torch.clamp(matrix - 1, min=0))
    return n_false_negatives


class _Metrics:
    """This class collects the metrics related to detection.

    Args:
        f_measure (float): F-measure of the model.
        precision (float): Precision of the model.
        recall (float): Recall of the model.
    """

    def __init__(self, f_measure: float, precision: float, recall: float):
        self.f_measure = f_measure
        self.precision = precision
        self.recall = recall


class _ResultCounters:
    """This class collects the number of prediction, TP and FN.

    Args:
        n_false_negatives (int): Number of false negatives.
        n_true (int): Number of true positives.
        n_predictions (int): Number of predictions.
    """

    def __init__(self, n_false_negatives: int, n_true: int, n_predicted: int):
        self.n_false_negatives = n_false_negatives
        self.n_true = n_true
        self.n_predicted = n_predicted

    def calculate_f_measure(self) -> _Metrics:
        """Calculates and returns precision, recall, and f-measure.

        Returns:
            _Metrics: _Metrics object with Precision, recall, and f-measure.
        """
        n_true_positives = self.n_true - self.n_false_negatives

        if self.n_predicted == 0:
            precision = 1.0
            recall = 0.0
        elif self.n_true == 0:
            precision = 0.0
            recall = 1.0
        else:
            precision = n_true_positives / self.n_predicted
            recall = n_true_positives / self.n_true

        f_measure = (2 * precision * recall) / (precision + recall + np.finfo(float).eps)
        return _Metrics(f_measure, precision, recall)


class _AggregatedResults:
    """This class collects the aggregated results for F-measure.

    The result contains:
        - f_measure_curve
        - precision_curve
        - recall_curve
        - all_classes_f_measure_curve
        - best_f_measure
        - best_threshold

    Args:
        classes (list[str]): list of classes.
    """

    def __init__(self, classes: list[str]):
        self.f_measure_curve: dict[str, list[float]] = {class_name: [] for class_name in classes}
        self.precision_curve: dict[str, list[float]] = {class_name: [] for class_name in classes}
        self.recall_curve: dict[str, list[float]] = {class_name: [] for class_name in classes}
        self.all_classes_f_measure_curve: list[float] = []
        self.best_f_measure: float = 0.0
        self.best_threshold: float = 0.0


class _OverallResults:
    """This class collects the overall results that is computed by the F-measure performance provider.

    Args:
        per_confidence (_AggregatedResults): _AggregatedResults object for each confidence level.
        best_f_measure_per_class (dict[str, float]): Best f-measure per class.
        best_f_measure (float): Best f-measure.
    """

    def __init__(
        self,
        per_confidence: _AggregatedResults,
        best_f_measure_per_class: dict[str, float],
        best_f_measure: float,
    ):
        self.per_confidence = per_confidence
        self.best_f_measure_per_class = best_f_measure_per_class
        self.best_f_measure = best_f_measure


class _FMeasureCalculator:
    """This class contains the functions to calculate FMeasure.

    Args:
       classes (list[str]): List of classes.
    """

    def __init__(self, classes: list[str]):
        self.classes = classes
        self.confidence_range = [0.025, 1.0, 0.025]
        self.nms_range = [0.1, 1, 0.05]
        self.default_confidence_threshold = 0.35

    def evaluate_detections(
        self,
        gt_entities: list[DetDataEntity],
        pred_entities: list[DetPredEntity],
        iou_threshold: float = 0.5,
    ) -> _OverallResults:
        """Evaluates detections by computing f_measures across multiple confidence thresholds and iou thresholds.

        By default, this function evaluates 39 confidence thresholds, finds the best confidence threshold and appends
        it to the result dict
        Each one of the (default 39+20) pairs of confidence and nms thresholds is used to evaluate the f-measure for
        each class, then the intermediate metrics are summed across classes to compute an all_classes f_measure.
        Finally, the best results across all evaluations are appended to the result dictionary along with the thresholds
        used to achieve them.

        Args:
            gt_entities (list[DetDataEntity]): List of ground truth entities.
            pred_entities (list[DetPredEntity]): List of predicted entities.
            iou_threshold (float): IOU threshold. Defaults to 0.5.

        Returns:
            _OverallResults: _OverallResults object with the result statistics (e.g F-measure).
        """
        best_f_measure_per_class = {}

        results_per_confidence = self._get_results_per_confidence(
            classes=self.classes.copy(),
            gt_entities=gt_entities,
            pred_entities=pred_entities,
            confidence_range=self.confidence_range,
            iou_threshold=iou_threshold,
        )

        best_f_measure = results_per_confidence.best_f_measure

        for class_name in self.classes:
            best_f_measure_per_class[class_name] = max(results_per_confidence.f_measure_curve[class_name])

        return _OverallResults(
            results_per_confidence,
            best_f_measure_per_class,
            best_f_measure,
        )

    def _get_results_per_confidence(
        self,
        classes: list[str],
        gt_entities: list[DetDataEntity],
        pred_entities: list[DetPredEntity],
        confidence_range: list[float],
        iou_threshold: float,
    ) -> _AggregatedResults:
        """Returns the results for confidence threshold in range confidence_range.

        Varies confidence based on confidence_range, the results are appended in a dictionary and returned, it also
        returns the best f_measure found and the confidence threshold used to get said f_measure

        Args:
            classes (list[str]): Names of classes to be evaluated.
            gt_entities (list[DetDataEntity]): List of ground truth entities.
            pred_entities (list[DetPredEntity]): List of predicted entities.
            confidence_range (list[float]): list of confidence thresholds to be evaluated.
            iou_threshold (float): IoU threshold to use for false negatives.

        Returns:
            _AggregatedResults: _AggregatedResults object with the result statistics (e.g F-measure).
        """
        result = _AggregatedResults(classes)
        result.best_threshold = 0.1

        for confidence_threshold in np.arange(*confidence_range):
            result_point = self.evaluate_classes(
                gt_entities=gt_entities,
                pred_entities=pred_entities,
                classes=classes,
                iou_threshold=iou_threshold,
                confidence_threshold=confidence_threshold,
            )
            all_classes_f_measure = result_point[ALL_CLASSES_NAME].f_measure
            result.all_classes_f_measure_curve.append(all_classes_f_measure)

            for class_name in classes:
                result.f_measure_curve[class_name].append(result_point[class_name].f_measure)
                result.precision_curve[class_name].append(result_point[class_name].precision)
                result.recall_curve[class_name].append(result_point[class_name].recall)
            if all_classes_f_measure > 0.0 and all_classes_f_measure >= result.best_f_measure:
                result.best_f_measure = all_classes_f_measure
                result.best_threshold = confidence_threshold
        return result

    def evaluate_classes(
        self,
        gt_entities: list[DetDataEntity],
        pred_entities: list[DetPredEntity],
        classes: list[str],
        iou_threshold: float,
        confidence_threshold: float,
    ) -> dict[str, _Metrics]:
        """Returns dict of f_measure, precision and recall for each class.

        Args:
            gt_entites (list[DetDataEntity]): List of ground truth entities.
            pred_entities (list[DetPredEntity]): List of predicted entities.
            classes (list[str]): list of classes to be evaluated.
            iou_threshold (float): IoU threshold to use for false negatives.
            confidence_threshold (float): Confidence threshold to use for false negatives.

        Returns:
            dict[str, _Metrics]: The metrics (e.g. F-measure) for each class.
        """
        result: dict[str, _Metrics] = {}

        all_classes_counters = _ResultCounters(0, 0, 0)

        if ALL_CLASSES_NAME in classes:
            classes.remove(ALL_CLASSES_NAME)
        for label_idx, class_name in enumerate(classes):
            metrics, counters = self.get_f_measure_for_class(
                gt_entities=gt_entities,
                pred_entities=pred_entities,
                label_idx=label_idx,
                iou_threshold=iou_threshold,
                confidence_threshold=confidence_threshold,
            )
            result[class_name] = metrics
            all_classes_counters.n_false_negatives += counters.n_false_negatives
            all_classes_counters.n_true += counters.n_true
            all_classes_counters.n_predicted += counters.n_predicted

        # for all classes
        result[ALL_CLASSES_NAME] = all_classes_counters.calculate_f_measure()
        return result

    def get_f_measure_for_class(
        self,
        gt_entities: list[DetDataEntity],
        pred_entities: list[DetPredEntity],
        label_idx: int,
        iou_threshold: float,
        confidence_threshold: float,
    ) -> tuple[_Metrics, _ResultCounters]:
        """Get f_measure for specific class, iou threshold, and confidence threshold.

        In order to reduce the number of redundant iterations and allow for cleaner, more general code later on,
        all boxes are filtered at this stage by class and predicted boxes are filtered by confidence threshold

        Args:
            gt_entities (list[DetDataEntity]): List of ground truth entities.
            pred_entities (list[DetPredEntity]): List of predicted entities.
            label_idx (int): Index of the class for which the boxes are filtered.
            iou_threshold (float): IoU threshold
            confidence_threshold (float): Confidence threshold

        Returns:
            tuple[_Metrics, _ResultCounters]: a structure containing the statistics (e.g. f_measure) and a structure
            containing the intermediated counters used to derive the stats (e.g. num. false positives)
        """
        batch_gt_bboxes = self.__filter_gt(
            gt_entities,
            label_idx=label_idx,
        )
        batch_pred_bboxes = self.__filter_pred(
            pred_entities,
            label_idx=label_idx,
            confidence_threshold=confidence_threshold,
        )

        if len(batch_gt_bboxes) > 0:
            result_counters = self.get_counters(
                batch_gt=batch_gt_bboxes,
                batch_pred=batch_pred_bboxes,
                iou_threshold=iou_threshold,
            )
            result_metrics = result_counters.calculate_f_measure()
            results = (result_metrics, result_counters)
        else:
            logger.warning("No ground truth images supplied for f-measure calculation.")
            # [f_measure, precision, recall, n_false_negatives, n_true, n_predicted]
            results = (_Metrics(0.0, 0.0, 0.0), _ResultCounters(0, 0, 0))
        return results

    @staticmethod
    def __filter_gt(
        entities: list[DetDataEntity],
        label_idx: int,
    ) -> list[Tensor]:
        """Filters boxes to only keep members of one class.

        Args:
            entities (list[DetDataEntity]): a list of DetDataEntity objects containing the ground truth annotations.
            label_idx (int): Index of the class for which the boxes are filtered.

        Returns:
            list[Tensor]: a list of bounding boxes for label_idx
        """
        batch_bboxes = []
        for entity in entities:
            keep = entity.labels == label_idx
            batch_bboxes.append(entity.bboxes[keep])
        return batch_bboxes

    @staticmethod
    def __filter_pred(
        entities: list[DetPredEntity],
        label_idx: int,
        confidence_threshold: float,
    ) -> list[Tensor]:
        """Filters boxes to only keep members of one class.

        Args:
            entities (list[DetPredEntity]): a list of DetPredEntity objects containing the predicted boxes.
            label_idx (int): Index of the class for which the boxes are filtered.
            confidence_threshold (float): Confidence threshold

        Returns:
            list[list[tuple]]: a list of lists of boxes
        """
        batch_bboxes = []
        for entity in entities:
            keep = (entity.labels == label_idx) & (entity.score > confidence_threshold)
            batch_bboxes.append(entity.bboxes[keep])
        return batch_bboxes

    @staticmethod
    def get_counters(
        batch_gt: list[Tensor],
        batch_pred: list[Tensor],
        iou_threshold: float,
    ) -> _ResultCounters:
        """Return counts of true positives, false positives and false negatives for a given iou threshold.

        For each image (the loop), compute the number of false negatives, the number of predicted boxes, and the number
        of ground truth boxes, then add each value to its corresponding counter

        Args:
            batch_gt (list[Tensor]): List of ground truth boxes
            batch_pred (list[Tensor]): List of predicted boxes
            iou_threshold (float): IoU threshold

        Returns:
            _ResultCounters: Structure containing the number of false negatives, true positives and predictions.
        """
        n_false_negatives = 0
        n_true = 0
        n_predicted = 0
        for gt_bboxes, pred_bboxes in zip(batch_gt, batch_pred, strict=True):
            n_true += len(gt_bboxes)
            n_predicted += len(pred_bboxes)
            if len(pred_bboxes) > 0:
                if len(gt_bboxes) > 0:
                    iou_matrix = box_iou(gt_bboxes, pred_bboxes)
                    n_false_negatives += get_n_false_negatives(iou_matrix, iou_threshold)
            else:
                n_false_negatives += len(gt_bboxes)
        return _ResultCounters(n_false_negatives, n_true, n_predicted)


class FMeasure(Metric):
    """Computes the f-measure (also known as F1-score) for a resultset.

    The f-measure is typically used in detection (localization) tasks to obtain a single number that balances precision
    and recall.

    To determine whether a predicted box matches a ground truth box an overlap measured
    is used based on a minimum
    intersection-over-union (IoU), by default a value of 0.5 is used.

    # TODO(someone): need to update for distriubted training. refer https://lightning.ai/docs/torchmetrics/stable/pages/implement.html

    Args:
        label_info (int): Dataclass including label information.
    """

    def __init__(
        self,
        label_info: LabelInfo,
    ):
        super().__init__()
        self.label_info: LabelInfo = label_info
        self._f_measure_per_confidence: dict | None = None
        self._best_confidence_threshold: float | None = None
        self._f_measure = float("-inf")

        self.reset()

    def reset(self) -> None:
        """Reset for every validation and test epoch.

        Please be careful that some variables should not be reset for each epoch.
        """
        super().reset()
        self.preds: list[DetPredEntity] = []
        self.targets: list[DetDataEntity] = []

    def update(self, preds: list[dict[str, Tensor]], target: list[dict[str, Tensor]]) -> None:
        """Update total predictions and targets from given batch predicitons and targets."""
        for i, (pred, tget) in enumerate(zip(preds, target)):
            self.preds.append(
                DetPredEntity(
                    image=tv_tensors.Image(torch.empty(0, 0)),
                    img_info=ImageInfo(img_idx=i, img_shape=(0, 0), ori_shape=(0, 0)),
                    bboxes=pred["boxes"],
                    score=pred["scores"],
                    labels=pred["labels"],
                ),
            )
            self.targets.append(
                DetDataEntity(
                    image=tv_tensors.Image(torch.empty(0, 0)),
                    img_info=ImageInfo(img_idx=i, img_shape=(0, 0), ori_shape=(0, 0)),
                    bboxes=tget["boxes"],
                    labels=tget["labels"],
                ),
            )

    def compute(self, best_confidence_threshold: float | None = None) -> dict:
        """Compute f1 score metric.

        Args:
            best_confidence_threshold (float | None): Pre-defined best confidence threshold.
                If this value is None, then FMeasure will find best confidence threshold and
                store it as member variable. Defaults to None.
        """
        boxes_pair = _FMeasureCalculator(classes=self.classes)
        result = boxes_pair.evaluate_detections(self.targets, self.preds)
        self._f_measure_per_label = {label: result.best_f_measure_per_class[label] for label in self.classes}

        if best_confidence_threshold is not None:
            (index,) = np.where(
                np.isclose(list(np.arange(*boxes_pair.confidence_range)), best_confidence_threshold),
            )
            computed_f_measure = result.per_confidence.all_classes_f_measure_curve[int(index)]
        else:
            self._f_measure_per_confidence = {
                "xs": list(np.arange(*boxes_pair.confidence_range)),
                "ys": result.per_confidence.all_classes_f_measure_curve,
            }
            computed_f_measure = result.best_f_measure
            best_confidence_threshold = result.per_confidence.best_threshold

        # TODO(jaegukhyun): There was no reset() function in this metric
        # There are some variables dependent on the best F1 metric, e.g., best_confidence_threshold
        # Now we added reset() function and revise some mechanism about it. However,
        # It is still unsure that it is correctly working with the implemented reset function.
        # Need to revisit. See other metric implement and this to learn how they work
        # https://github.com/Lightning-AI/torchmetrics/blob/v1.2.1/src/torchmetrics/metric.py
        if self._f_measure < computed_f_measure:
            self._f_measure = result.best_f_measure
            self._best_confidence_threshold = best_confidence_threshold
        return {"f1-score": Tensor([computed_f_measure])}

    @property
    def f_measure(self) -> float:
        """Returns the f-measure."""
        return self._f_measure

    @property
    def f_measure_per_label(self) -> dict[str, float]:
        """Returns the f-measure per label as dictionary (Label -> Score)."""
        return self._f_measure_per_label

    @property
    def f_measure_per_confidence(self) -> None | dict:
        """Returns the curve for f-measure per confidence as dictionary if exists."""
        return self._f_measure_per_confidence

    @property
    def best_confidence_threshold(self) -> float:
        """Returns best confidence threshold as ScoreMetric if exists."""
        if self._best_confidence_threshold is None:
            msg = (
                "Cannot obtain best_confidence_threshold updated previously. "
                "Please execute self.update(best_confidence_threshold=None) first."
            )
            raise RuntimeError(msg)
        return self._best_confidence_threshold

    @property
    def classes(self) -> list[str]:
        """Class information of dataset."""
        return self.label_info.label_names


class MeanAveragePrecisionFMeasure(MetricCollection):
    """Computes the mean AP with f-measure for a resultset.

    NOTE: IMPORTANT!!! Do not use this metric to evaluate a F1 score on a test set.
        This is because it can pollute test evaluation.
        It will optimize the confidence threshold on the test set by
        doing line search on confidence threshold axis.
        The correct way to obtain the test set F1 score is to use
        the best confidence threshold obtained from the validation set.
        You should use `--metric otx.core.metrics.fmeasure.FMeasureCallable` override
        to correctly obtain F1 score from a test set.
    """

    def __init__(self, box_format: str, iou_type: str, label_info: LabelInfo, **kwargs):
        map_kwargs = self._filter_kwargs(MeanAveragePrecision, kwargs)
        fmeasure_kwargs = self._filter_kwargs(FMeasure, kwargs)

        super().__init__(
            [
                MeanAveragePrecision(box_format, iou_type, **map_kwargs),
                FMeasure(label_info, **fmeasure_kwargs),
            ],
        )

    def _filter_kwargs(self, cls: type[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        cls_params = inspect.signature(cls.__init__).parameters
        valid_keys = set(cls_params.keys()) - {"self"}
        return {k: v for k, v in kwargs.items() if k in valid_keys}


def _f_measure_callable(label_info: LabelInfo) -> FMeasure:
    return FMeasure(label_info=label_info)


def _mean_ap_f_measure_callable(label_info: LabelInfo) -> MeanAveragePrecisionFMeasure:
    return MeanAveragePrecisionFMeasure(
        box_format="xyxy",
        iou_type="bbox",
        label_info=label_info,
    )


FMeasureCallable = _f_measure_callable

MeanAveragePrecisionFMeasureCallable = _mean_ap_f_measure_callable
