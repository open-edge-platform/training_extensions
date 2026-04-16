# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from getitune.data import DetectionDataset, InstanceSegDataset, MulticlassClsDataset, MultilabelClsDataset
from getitune.metrics.accuracy import MultiClassClsMetricCallable, MultiLabelClsMetricCallable
from getitune.metrics.mean_ap import MaskRLEMeanAPCallable, MeanAPCallable
from getitune.metrics.types import MetricCallable
from getitune.types.task import TaskType as GetiTuneTaskType

from app.execution.common.getitune_converters import (
    get_getitune_dataset_class_by_task_type,
    get_getitune_task_type_by_task,
    get_metric_by_task,
)
from app.models import Task, TaskType


class TestGetOtxTaskTypeByTask:
    @pytest.mark.parametrize(
        "task_type,exclusive_labels,expected",
        [
            (TaskType.CLASSIFICATION, True, GetiTuneTaskType.MULTI_CLASS_CLS),
            (TaskType.CLASSIFICATION, False, GetiTuneTaskType.MULTI_LABEL_CLS),
            (TaskType.DETECTION, False, GetiTuneTaskType.DETECTION),
            (TaskType.INSTANCE_SEGMENTATION, False, GetiTuneTaskType.INSTANCE_SEGMENTATION),
        ],
        ids=["multiclass_cls", "multilabel_cls", "detection", "instance_seg"],
    )
    def test_supported_task_types(self, task_type: TaskType, exclusive_labels: bool, expected: GetiTuneTaskType):
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
            (GetiTuneTaskType.MULTI_CLASS_CLS, MulticlassClsDataset),
            (GetiTuneTaskType.MULTI_LABEL_CLS, MultilabelClsDataset),
            (GetiTuneTaskType.DETECTION, DetectionDataset),
            (GetiTuneTaskType.INSTANCE_SEGMENTATION, InstanceSegDataset),
        ],
        ids=["multiclass_cls", "multilabel_cls", "detection", "instance_seg"],
    )
    def test_supported_getitune_task_types(self, getitune_task_type: GetiTuneTaskType, expected_class: type):
        assert get_getitune_dataset_class_by_task_type(getitune_task_type) is expected_class

    @pytest.mark.parametrize(
        "getitune_task_type",
        [
            GetiTuneTaskType.SEMANTIC_SEGMENTATION,
            GetiTuneTaskType.H_LABEL_CLS,
            GetiTuneTaskType.ROTATED_DETECTION,
            GetiTuneTaskType.KEYPOINT_DETECTION,
        ],
        ids=["semantic_seg", "h_label_cls", "rotated_det", "keypoint_det"],
    )
    def test_unsupported_getitune_task_type_raises(self, getitune_task_type: GetiTuneTaskType):
        """A getitune task type without a mapped dataset class must raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported getitune task type"):
            get_getitune_dataset_class_by_task_type(getitune_task_type)
