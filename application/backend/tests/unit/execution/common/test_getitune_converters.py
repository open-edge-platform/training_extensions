# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from getitune.data import DetectionDataset, InstanceSegDataset, MulticlassClsDataset, MultilabelClsDataset
from getitune.metrics.accuracy import MultiClassClsMetricCallable, MultiLabelClsMetricCallable
from getitune.metrics.mean_ap import MaskRLEMeanAPCallable, MeanAPCallable
from getitune.metrics.types import MetricCallable
from getitune.types.task import TaskType

from app.execution.common.getitune_converters import (
    get_metric_by_task,
    get_getitune_dataset_class_by_task_type,
    get_getitune_task_type_by_task,
)
from app.models import Task, TaskType


class TestGetOtxTaskTypeByTask:
    @pytest.mark.parametrize(
        "task_type,exclusive_labels,expected",
        [
            (TaskType.CLASSIFICATION, True, TaskType.MULTI_CLASS_CLS),
            (TaskType.CLASSIFICATION, False, TaskType.MULTI_LABEL_CLS),
            (TaskType.DETECTION, False, TaskType.DETECTION),
            (TaskType.INSTANCE_SEGMENTATION, False, TaskType.INSTANCE_SEGMENTATION),
        ],
        ids=["multiclass_cls", "multilabel_cls", "detection", "instance_seg"],
    )
    def test_supported_task_types(self, task_type: TaskType, exclusive_labels: bool, expected: TaskType):
        task = Task(task_type=task_type, exclusive_labels=exclusive_labels)
        assert get_getitune_task_type_by_task(task) == expected

    def test_unsupported_task_type_raises(self):
        """An unknown task type must raise ValueError."""
        task = Mock(spec=Task)
        task.task_type = "semantic_segmentation"
        with pytest.raises(ValueError, match="Unsupported task type"):
            get_getitune_task_type_by_task(task)


class TestGetMetricByTask:
    @pytest.mark.parametrize(
        "task_type,exclusive_labels,expected",
        [
            (TaskType.CLASSIFICATION, True, MultiClassClsMetricCallable),
            (TaskType.CLASSIFICATION, False, MultiLabelClsMetricCallable),
            (TaskType.DETECTION, False, MeanAPCallable),
            (TaskType.INSTANCE_SEGMENTATION, False, MaskRLEMeanAPCallable),
        ],
        ids=["multiclass_cls", "multilabel_cls", "detection", "instance_seg"],
    )
    def test_supported_task_types(self, task_type: TaskType, exclusive_labels: bool, expected: MetricCallable):
        task = Task(task_type=task_type, exclusive_labels=exclusive_labels)
        assert get_metric_by_task(task) is expected

    def test_unsupported_task_type_raises(self):
        """An unknown task type must raise ValueError."""
        task = Mock(spec=Task)
        task.task_type = "semantic_segmentation"
        with pytest.raises(ValueError, match="Unsupported task type"):
            get_metric_by_task(task)


class TestGetOtxDatasetClassByTaskType:
    @pytest.mark.parametrize(
        "getitune_task_type,expected_class",
        [
            (TaskType.MULTI_CLASS_CLS, MulticlassClsDataset),
            (TaskType.MULTI_LABEL_CLS, MultilabelClsDataset),
            (TaskType.DETECTION, DetectionDataset),
            (TaskType.INSTANCE_SEGMENTATION, InstanceSegDataset),
        ],
        ids=["multiclass_cls", "multilabel_cls", "detection", "instance_seg"],
    )
    def test_supported_otx_task_types(self, getitune_task_type: TaskType, expected_class: type):
        assert get_getitune_dataset_class_by_task_type(getitune_task_type) is expected_class

    @pytest.mark.parametrize(
        "getitune_task_type",
        [
            TaskType.SEMANTIC_SEGMENTATION,
            TaskType.H_LABEL_CLS,
            TaskType.ROTATED_DETECTION,
            TaskType.KEYPOINT_DETECTION,
        ],
        ids=["semantic_seg", "h_label_cls", "rotated_det", "keypoint_det"],
    )
    def test_unsupported_otx_task_type_raises(self, getitune_task_type: TaskType):
        """A Geti Tune task type without a mapped dataset class must raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported Geti Tune task type"):
            get_getitune_dataset_class_by_task_type(getitune_task_type)
