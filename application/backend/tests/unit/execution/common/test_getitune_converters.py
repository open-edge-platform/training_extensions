# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import numpy as np
import pytest
import torch
from getitune.data import DetectionDataset, InstanceSegDataset, MulticlassClsDataset, MultilabelClsDataset
from getitune.metrics.accuracy import MultiClassClsMetricCallable, MultiLabelClsMetricCallable
from getitune.metrics.mean_ap import MaskRLEMeanAPCallable, MeanAPCallable
from getitune.metrics.types import MetricCallable
from getitune.types.task import TaskType as GetiTuneTaskType

from app.execution.common.getitune_converters import (
    convert_metrics,
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


class TestConvertMetrics:
    def test_empty_dict(self):
        assert convert_metrics({}) == {}

    def test_strips_namespace_prefix(self):
        """Keys containing '/' must keep only the suffix after the first '/'."""
        result = convert_metrics({"val/accuracy": 0.9})
        assert result == {"accuracy": pytest.approx(0.9)}

    def test_keeps_key_without_prefix(self):
        result = convert_metrics({"accuracy": 0.5})
        assert result == {"accuracy": pytest.approx(0.5)}

    def test_scalar_tensor_is_converted(self):
        result = convert_metrics({"val/loss": torch.tensor(1.25)})
        assert result == {"loss": pytest.approx(1.25)}

    def test_non_scalar_tensor_is_skipped(self):
        """Multi-element tensors (e.g., per-class metrics) must be skipped."""
        result = convert_metrics({"val/per_class": torch.tensor([0.1, 0.2, 0.3])})
        assert result == {}

    def test_scalar_numpy_array_is_converted(self):
        result = convert_metrics({"val/score": np.array(0.75)})
        assert result == {"score": pytest.approx(0.75)}

    def test_single_element_numpy_array_is_converted(self):
        result = convert_metrics({"val/score": np.array([0.75])})
        assert result == {"score": pytest.approx(0.75)}

    def test_non_scalar_numpy_array_is_skipped(self):
        result = convert_metrics({"val/per_class": np.array([0.1, 0.2])})
        assert result == {}

    @pytest.mark.parametrize(
        "value",
        [
            [1, 2, 3],
            (1, 2),
            {"a": 1},
            [torch.tensor([[1, 0], [0, 1]])],  # multilabel conf_matrix shape
        ],
        ids=["list", "tuple", "dict", "list_of_tensors"],
    )
    def test_aggregate_containers_are_skipped(self, value):
        """Lists/tuples/dicts (e.g., multilabel 'conf_matrix') must be skipped, not coerced."""
        result = convert_metrics({"val/conf_matrix": value})
        assert result == {}

    @pytest.mark.parametrize("value,expected", [(1, 1.0), (2.5, 2.5), ("3.14", 3.14), (True, 1.0)])
    def test_coercible_scalars(self, value, expected):
        result = convert_metrics({"val/x": value})
        assert result == {"x": pytest.approx(expected)}

    def test_non_numeric_value_is_skipped(self):
        """Values that cannot be converted to float must be skipped, not raise."""
        result = convert_metrics({"val/note": "not-a-number"})
        assert result == {}

    def test_none_value_is_skipped(self):
        result = convert_metrics({"val/x": None})
        assert result == {}

    def test_mixed_metrics(self):
        """Realistic multilabel-classification payload mixing scalars and a conf_matrix list."""
        metrics = {
            "val/accuracy": torch.tensor(0.8),
            "val/f1": 0.7,
            "val/conf_matrix": [torch.tensor([[5, 1], [0, 4]]), torch.tensor([[3, 2], [1, 4]])],
            "val/per_class": torch.tensor([0.6, 0.9]),
            "loss": np.float32(0.123),
        }
        result = convert_metrics(metrics)
        assert set(result.keys()) == {"accuracy", "f1", "loss"}
        assert result["accuracy"] == pytest.approx(0.8)
        assert result["f1"] == pytest.approx(0.7)
        assert result["loss"] == pytest.approx(0.123, rel=1e-5)

    def test_per_class_keys_preserve_class_name(self):
        """Per-class metric keys (val/metric/ClassName) must keep the class name after stripping the phase prefix."""
        metrics = {
            "val/precision/cat": 0.85,
            "val/recall/dog": 0.70,
            "val/map_50/Cabernet Franc": 0.90,
        }
        result = convert_metrics(metrics)
        assert result == {
            "precision/cat": pytest.approx(0.85),
            "recall/dog": pytest.approx(0.70),
            "map_50/Cabernet Franc": pytest.approx(0.90),
        }
