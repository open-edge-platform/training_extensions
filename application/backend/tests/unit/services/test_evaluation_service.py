# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np
import pytest
from datumaro.experimental.categories import LabelCategories
from datumaro.experimental.dataset import Dataset
from datumaro.experimental.fields import ImageInfo

from app.services.datumaro_converter import (
    ClassificationSample,
    DetectionSample,
    InstanceSegmentationSample,
    MultilabelClassificationSample,
)
from app.services.evaluation_service import (
    AveragingMethod,
    DetectionEvaluator,
    EvaluationService,
    InstanceSegmentationEvaluator,
    MultiClassClassificationEvaluator,
    MultiLabelClassificationEvaluator,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def fxt_multiclass_classification_dataset_gt() -> Dataset:
    # Ground truth dataset for multiclass classification task
    dataset = Dataset(ClassificationSample, categories={"label": LabelCategories(labels=("cat", "dog", "bird"))})
    img_info = ImageInfo(width=100, height=100)
    samples = (
        ClassificationSample(image="/dummy/path/A.jpg", image_info=img_info, label=0),
        ClassificationSample(image="/dummy/path/B.jpg", image_info=img_info, label=1),
        ClassificationSample(image="/dummy/path/C.jpg", image_info=img_info, label=2),
        ClassificationSample(image="/dummy/path/D.jpg", image_info=img_info, label=1),
        ClassificationSample(image="/dummy/path/E.jpg", image_info=img_info, label=2),
    )
    for sample in samples:
        dataset.append(sample)
    return dataset


@pytest.fixture
def fxt_multiclass_classification_dataset_pred() -> Dataset:
    # Prediction dataset for multiclass classification task
    dataset = Dataset(ClassificationSample, categories={"label": LabelCategories(labels=("cat", "dog", "bird"))})
    img_info = ImageInfo(width=100, height=100)
    samples = (
        ClassificationSample(image="/dummy/path/A.jpg", image_info=img_info, label=0),  # correct
        ClassificationSample(image="/dummy/path/B.jpg", image_info=img_info, label=2),  # wrong
        ClassificationSample(image="/dummy/path/C.jpg", image_info=img_info, label=1),  # wrong
        ClassificationSample(image="/dummy/path/D.jpg", image_info=img_info, label=1),  # correct
        ClassificationSample(image="/dummy/path/E.jpg", image_info=img_info, label=2),  # correct
    )
    for sample in samples:
        dataset.append(sample)
    return dataset


@pytest.fixture
def fxt_multilabel_classification_dataset_gt() -> Dataset:
    # Ground truth dataset for multilabel classification task
    dataset = Dataset(
        MultilabelClassificationSample, categories={"label": LabelCategories(labels=("pop", "rock", "jazz"))}
    )
    img_info = ImageInfo(width=100, height=100)
    samples = (
        MultilabelClassificationSample(image="/dummy/path/A.jpg", image_info=img_info, label=np.array([0, 1])),
        MultilabelClassificationSample(image="/dummy/path/B.jpg", image_info=img_info, label=np.array([1])),
        MultilabelClassificationSample(image="/dummy/path/C.jpg", image_info=img_info, label=np.array([2, 0])),
    )
    for sample in samples:
        dataset.append(sample)
    return dataset


@pytest.fixture
def fxt_multilabel_classification_dataset_pred() -> Dataset:
    # Prediction dataset for multilabel classification task
    dataset = Dataset(
        MultilabelClassificationSample, categories={"label": LabelCategories(labels=("pop", "rock", "jazz"))}
    )
    img_info = ImageInfo(width=100, height=100)
    samples = (
        MultilabelClassificationSample(
            image="/dummy/path/A.jpg", image_info=img_info, label=np.array([0])
        ),  # missing one label
        MultilabelClassificationSample(
            image="/dummy/path/B.jpg", image_info=img_info, label=np.array([1, 2])
        ),  # one extra label
        MultilabelClassificationSample(
            image="/dummy/path/C.jpg", image_info=img_info, label=np.array([2, 0])
        ),  # correct
    )
    for sample in samples:
        dataset.append(sample)
    return dataset


@pytest.fixture
def fxt_detection_dataset_gt() -> Dataset:
    # Ground truth dataset for detection task
    dataset = Dataset(DetectionSample, categories={"label": LabelCategories(labels=("car", "person"))})
    img_info = ImageInfo(width=100, height=100)
    samples = (
        DetectionSample(
            image="/dummy/path/A.jpg",
            image_info=img_info,
            bboxes=np.array([[10, 15, 30, 35]]),
            label=np.array([1]),
        ),
        DetectionSample(
            image="/dummy/path/B.jpg",
            image_info=img_info,
            bboxes=np.array([[5, 5, 20, 20], [25, 30, 50, 60]]),
            label=np.array([0, 1]),
        ),
        DetectionSample(
            image="/dummy/path/C.jpg",
            image_info=img_info,
            bboxes=np.array([[0, 0, 15, 15]]),
            label=np.array([0]),
        ),
    )
    for sample in samples:
        dataset.append(sample)
    return dataset


@pytest.fixture
def fxt_detection_dataset_pred() -> Dataset:
    # Prediction dataset for detection task
    dataset = Dataset(DetectionSample, categories={"label": LabelCategories(labels=("car", "person"))})
    img_info = ImageInfo(width=100, height=100)
    samples = (
        DetectionSample(
            image="/dummy/path/A.jpg",
            image_info=img_info,
            bboxes=np.array([[10, 20, 30, 40]]),  # partial overlap (IoU = 0.6)
            label=np.array([1]),  # correct
        ),
        DetectionSample(
            image="/dummy/path/B.jpg",
            image_info=img_info,
            bboxes=np.array([[5, 5, 20, 20], [25, 30, 50, 60]]),  # correct
            label=np.array([0, 1]),  # correct
        ),
        DetectionSample(
            image="/dummy/path/C.jpg",
            image_info=img_info,
            bboxes=np.array([[0, 0, 15, 15]]),  # correct
            label=np.array([1]),  # wrong
        ),
    )
    for sample in samples:
        dataset.append(sample)
    return dataset


@pytest.fixture
def fxt_instance_segmentation_dataset_gt() -> Dataset:
    # Ground truth dataset for instance segmentation task
    dataset = Dataset(InstanceSegmentationSample, categories={"label": LabelCategories(labels=("apple", "banana"))})
    img_info = ImageInfo(width=100, height=100)
    samples = (
        InstanceSegmentationSample(
            image="/dummy/path/A.jpg",
            image_info=img_info,
            polygons=np.array([[[10, 20], [30, 40], [40, 70], [10, 60]], [[10, 20], [30, 40], [50, 40]]], dtype=object),
            label=np.array([0, 1]),
        ),
        InstanceSegmentationSample(
            image="/dummy/path/B.jpg",
            image_info=img_info,
            polygons=np.array([[[50, 50], [90, 50], [50, 80]]]),
            label=np.array([0]),
        ),
        InstanceSegmentationSample(
            image="/dummy/path/C.jpg",
            image_info=img_info,
            polygons=np.array([[[15, 15], [25, 15], [25, 25], [15, 25]]]),
            label=np.array([1]),
        ),
    )
    for sample in samples:
        dataset.append(sample)
    return dataset


@pytest.fixture
def fxt_instance_segmentation_dataset_pred() -> Dataset:
    # Prediction dataset for instance segmentation task
    dataset = Dataset(InstanceSegmentationSample, categories={"label": LabelCategories(labels=("apple", "banana"))})
    img_info = ImageInfo(width=100, height=100)
    samples = (
        InstanceSegmentationSample(
            image="/dummy/path/A.jpg",
            image_info=img_info,
            polygons=np.array(
                [[[10, 20], [30, 40], [40, 70], [10, 60]], [[10, 20], [30, 40], [50, 40]]], dtype=object
            ),  # correct
            label=np.array([0, 1]),  # correct
        ),
        InstanceSegmentationSample(
            image="/dummy/path/B.jpg",
            image_info=img_info,
            polygons=np.array([[[50, 50], [82, 50], [50, 74]]]),  # partial overlap (64% IoU)
            label=np.array([0]),  # correct
        ),
        InstanceSegmentationSample(
            image="/dummy/path/C.jpg",
            image_info=img_info,
            polygons=np.array([[[15, 15], [25, 15], [25, 25], [15, 25]]]),  # correct
            label=np.array([0]),  # wrong
        ),
    )
    for sample in samples:
        dataset.append(sample)
    return dataset


class TestEvaluationService:
    @pytest.mark.parametrize(
        "task_type,gt_dataset_fixture,pred_dataset_fixture",
        [
            (
                "multiclass_classification",
                "fxt_multiclass_classification_dataset_gt",
                "fxt_multiclass_classification_dataset_pred",
            ),
            (
                "multilabel_classification",
                "fxt_multilabel_classification_dataset_gt",
                "fxt_multilabel_classification_dataset_pred",
            ),
            ("detection", "fxt_detection_dataset_gt", "fxt_detection_dataset_pred"),
            ("instance_segmentation", "fxt_instance_segmentation_dataset_gt", "fxt_instance_segmentation_dataset_pred"),
        ],
        ids=["multiclass_classification", "multilabel_classification", "detection", "instance_segmentation"],
    )
    def test_get_evaluator(self, request, task_type, gt_dataset_fixture, pred_dataset_fixture) -> None:
        ground_truth_dataset: Dataset = request.getfixturevalue(gt_dataset_fixture)
        prediction_dataset: Dataset = request.getfixturevalue(pred_dataset_fixture)

        # Retrieve evaluator for classification task
        evaluator = EvaluationService.get_evaluator(
            predictions_dataset=prediction_dataset, ground_truth_dataset=ground_truth_dataset
        )

        match task_type:
            case "multiclass_classification":
                assert isinstance(evaluator, MultiClassClassificationEvaluator)
            case "multilabel_classification":
                assert isinstance(evaluator, MultiLabelClassificationEvaluator)
            case "detection":
                assert isinstance(evaluator, DetectionEvaluator)
            case "instance_segmentation":
                assert isinstance(evaluator, InstanceSegmentationEvaluator)
            case _:
                raise AssertionError(f"Unrecognized task type {task_type}")


class TestMultiClassClassificationEvaluator:
    def test_evaluate_with_imperfect_predictions(
        self, fxt_multiclass_classification_dataset_gt, fxt_multiclass_classification_dataset_pred
    ) -> None:
        evaluator = MultiClassClassificationEvaluator(
            predictions_dataset=fxt_multiclass_classification_dataset_pred,
            ground_truth_dataset=fxt_multiclass_classification_dataset_gt,
        )

        precision_micro = evaluator.precision(averaging_method=AveragingMethod.MICRO)
        precision_macro = evaluator.precision(averaging_method=AveragingMethod.MACRO)
        precision_weighted = evaluator.precision(averaging_method=AveragingMethod.WEIGHTED)
        recall_micro = evaluator.recall(averaging_method=AveragingMethod.MICRO)
        recall_macro = evaluator.recall(averaging_method=AveragingMethod.MACRO)
        recall_weighted = evaluator.recall(averaging_method=AveragingMethod.WEIGHTED)
        f1_score_micro = evaluator.f1_score(averaging_method=AveragingMethod.MICRO)
        f1_score_macro = evaluator.f1_score(averaging_method=AveragingMethod.MACRO)
        f1_score_weighted = evaluator.f1_score(averaging_method=AveragingMethod.WEIGHTED)
        accuracy = evaluator.accuracy()
        confusion_matrix = evaluator.confusion_matrix()

        assert accuracy == pytest.approx(3 / 5)  # 3 correct out of 5
        # In multiclass classification, micro precision, recall, micro F1 and accuracy are equal
        assert precision_micro == recall_micro == f1_score_micro == accuracy
        assert precision_macro == pytest.approx(2 / 3)  # (1 + 0.5 + 0.5) / 3
        assert recall_macro == pytest.approx(2 / 3)
        assert f1_score_macro == pytest.approx(2 / 3)
        assert precision_weighted == pytest.approx(3 / 5)  # (1*1 + 0.5*2 + 0.5*2) / 5
        assert recall_weighted == pytest.approx(3 / 5)
        assert f1_score_weighted == pytest.approx(3 / 5)
        assert (confusion_matrix == np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1]])).all()

    def test_evaluate_with_perfect_predictions(self, fxt_multiclass_classification_dataset_gt) -> None:
        evaluator = MultiClassClassificationEvaluator(
            predictions_dataset=fxt_multiclass_classification_dataset_gt,  # using GT as predictions, hence perfect
            ground_truth_dataset=fxt_multiclass_classification_dataset_gt,
        )

        precision_micro = evaluator.precision(averaging_method=AveragingMethod.MICRO)
        precision_macro = evaluator.precision(averaging_method=AveragingMethod.MACRO)
        precision_weighted = evaluator.precision(averaging_method=AveragingMethod.WEIGHTED)
        recall_micro = evaluator.recall(averaging_method=AveragingMethod.MICRO)
        recall_macro = evaluator.recall(averaging_method=AveragingMethod.MACRO)
        recall_weighted = evaluator.recall(averaging_method=AveragingMethod.WEIGHTED)
        f1_score_micro = evaluator.f1_score(averaging_method=AveragingMethod.MICRO)
        f1_score_macro = evaluator.f1_score(averaging_method=AveragingMethod.MACRO)
        f1_score_weighted = evaluator.f1_score(averaging_method=AveragingMethod.WEIGHTED)
        accuracy = evaluator.accuracy()
        confusion_matrix = evaluator.confusion_matrix()

        assert accuracy == 1.0
        assert precision_micro == precision_macro == precision_weighted == 1.0
        assert recall_micro == recall_macro == recall_weighted == 1.0
        assert f1_score_micro == f1_score_macro == f1_score_weighted == 1.0
        assert np.trace(confusion_matrix) == confusion_matrix.sum()  # all non-zero values are on the diagonal


class TestMultiLabelClassificationEvaluator:
    def test_evaluate_with_imperfect_predictions(
        self, fxt_multilabel_classification_dataset_gt, fxt_multilabel_classification_dataset_pred
    ) -> None:
        evaluator = MultiLabelClassificationEvaluator(
            predictions_dataset=fxt_multilabel_classification_dataset_pred,
            ground_truth_dataset=fxt_multilabel_classification_dataset_gt,
        )

        precision_micro = evaluator.precision(averaging_method=AveragingMethod.MICRO)
        precision_macro = evaluator.precision(averaging_method=AveragingMethod.MACRO)
        precision_weighted = evaluator.precision(averaging_method=AveragingMethod.WEIGHTED)
        recall_micro = evaluator.recall(averaging_method=AveragingMethod.MICRO)
        recall_macro = evaluator.recall(averaging_method=AveragingMethod.MACRO)
        recall_weighted = evaluator.recall(averaging_method=AveragingMethod.WEIGHTED)
        f1_score_micro = evaluator.f1_score(averaging_method=AveragingMethod.MICRO)
        f1_score_macro = evaluator.f1_score(averaging_method=AveragingMethod.MACRO)
        f1_score_weighted = evaluator.f1_score(averaging_method=AveragingMethod.WEIGHTED)
        accuracy = evaluator.accuracy()

        assert accuracy == pytest.approx(1 / 3)  # only one sample is completely correct (subset accuracy)
        assert precision_micro == pytest.approx(4 / 5)  # 4 correct labels out of 5 predicted
        assert precision_macro == pytest.approx(5 / 6)  # (2/2 + 1/1 + 1/2) / 3
        assert precision_weighted == pytest.approx(9 / 10)  # (1*2 + 1*2 + 0.5*0.5) / 5
        assert recall_micro == pytest.approx(4 / 5)  # 4 ground truth labels out of 5 found
        assert recall_macro == pytest.approx(5 / 6)  # (2/2 + 1/2 + 1/1) / 3
        assert recall_weighted == recall_micro  # always identical to micro recall
        assert f1_score_micro == pytest.approx(4 / 5)  # harmonic mean of micro precision and recall
        assert f1_score_macro == pytest.approx(7 / 9)  # harmonic mean of macro precision and recall
        assert f1_score_weighted == pytest.approx(4 / 5)  # harmonic mean of weighted precision and recall

    def test_evaluate_with_perfect_predictions(self, fxt_multilabel_classification_dataset_gt) -> None:
        evaluator = MultiLabelClassificationEvaluator(
            predictions_dataset=fxt_multilabel_classification_dataset_gt,  # using GT as predictions, hence perfect
            ground_truth_dataset=fxt_multilabel_classification_dataset_gt,
        )

        precision_micro = evaluator.precision(averaging_method=AveragingMethod.MICRO)
        precision_macro = evaluator.precision(averaging_method=AveragingMethod.MACRO)
        precision_weighted = evaluator.precision(averaging_method=AveragingMethod.WEIGHTED)
        recall_micro = evaluator.recall(averaging_method=AveragingMethod.MICRO)
        recall_macro = evaluator.recall(averaging_method=AveragingMethod.MACRO)
        recall_weighted = evaluator.recall(averaging_method=AveragingMethod.WEIGHTED)
        f1_score_micro = evaluator.f1_score(averaging_method=AveragingMethod.MICRO)
        f1_score_macro = evaluator.f1_score(averaging_method=AveragingMethod.MACRO)
        f1_score_weighted = evaluator.f1_score(averaging_method=AveragingMethod.WEIGHTED)
        accuracy = evaluator.accuracy()

        assert accuracy == 1.0
        assert precision_micro == precision_macro == precision_weighted == 1.0
        assert recall_micro == recall_macro == recall_weighted == 1.0
        assert f1_score_micro == f1_score_macro == f1_score_weighted == 1.0


class TestDetectionEvaluator:
    def test_evaluate_with_imperfect_predictions(self, fxt_detection_dataset_gt, fxt_detection_dataset_pred) -> None:
        """Evaluate on a scenario where some predictions are correct and some are incorrect."""
        evaluator = DetectionEvaluator(
            predictions_dataset=fxt_detection_dataset_pred,
            ground_truth_dataset=fxt_detection_dataset_gt,
        )

        map_dict = evaluator.mean_average_precision()

        # Example is constructed with a bbox with 60% IoU, that's between the 50% and 75% thresholds
        assert 0 < map_dict["AP_75"] < map_dict["AP_50"] < 1
        assert 0 < map_dict["AP_all"] < 1

    def test_evaluate_with_perfect_predictions(self, fxt_detection_dataset_gt) -> None:
        """Evaluate on a scenario where all predictions are correct."""
        evaluator = DetectionEvaluator(
            predictions_dataset=fxt_detection_dataset_gt,  # using GT as predictions, hence perfect
            ground_truth_dataset=fxt_detection_dataset_gt,
        )

        map_dict = evaluator.mean_average_precision()

        assert map_dict["AP_50"] == map_dict["AP_75"] == map_dict["AP_all"] == 1.0


class TestInstanceSegmentationEvaluator:
    def test_evaluate_with_imperfect_predictions(
        self, fxt_instance_segmentation_dataset_gt, fxt_instance_segmentation_dataset_pred
    ) -> None:
        """Evaluate on a scenario where some predictions are correct and some are incorrect."""
        evaluator = InstanceSegmentationEvaluator(
            predictions_dataset=fxt_instance_segmentation_dataset_pred,
            ground_truth_dataset=fxt_instance_segmentation_dataset_gt,
        )

        map_dict = evaluator.mean_average_precision()

        # Example is constructed with a polygon with 64% IoU, that's between the 50% and 75% thresholds
        assert 0 < map_dict["AP_75"] < map_dict["AP_50"] < 1
        assert 0 < map_dict["AP_all"] < 1

    def test_evaluate_with_perfect_predictions(self, fxt_instance_segmentation_dataset_gt) -> None:
        """Evaluate on a scenario where all predictions are correct."""
        evaluator = InstanceSegmentationEvaluator(
            predictions_dataset=fxt_instance_segmentation_dataset_gt,  # using GT as predictions, hence perfect
            ground_truth_dataset=fxt_instance_segmentation_dataset_gt,
        )

        map_dict = evaluator.mean_average_precision()

        assert map_dict["AP_50"] == map_dict["AP_75"] == map_dict["AP_all"] == 1.0
