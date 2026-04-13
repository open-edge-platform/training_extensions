# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import torch
from loguru import logger
from getitune import OTXTaskType
from getitune.data import OTXDetectionDataset, OTXInstanceSegDataset, OTXMulticlassClsDataset, OTXMultilabelClsDataset
from getitune.data.dataset.base import OTXDataset
from getitune.metrics import MetricCallable
from getitune.metrics.accuracy import MultiClassClsMetricCallable, MultiLabelClsMetricCallable
from getitune.metrics.mean_ap import MaskRLEMeanAPCallable, MeanAPCallable

from app.models import Task, TaskType


def get_otx_task_type_by_task(task: Task) -> OTXTaskType:
    """Map internal Task to OTXTaskType."""
    match task.task_type:
        case TaskType.CLASSIFICATION:
            if task.exclusive_labels:
                return OTXTaskType.MULTI_CLASS_CLS
            return OTXTaskType.MULTI_LABEL_CLS
        case TaskType.DETECTION:
            return OTXTaskType.DETECTION
        case TaskType.INSTANCE_SEGMENTATION:
            return OTXTaskType.INSTANCE_SEGMENTATION
        case _:
            raise ValueError(f"Unsupported task type: {task.task_type}")


def get_metric_by_task(task: Task) -> MetricCallable:
    """Map internal Task to Metric."""
    match task.task_type:
        case TaskType.CLASSIFICATION:
            if task.exclusive_labels:
                return MultiClassClsMetricCallable
            return MultiLabelClsMetricCallable
        case TaskType.DETECTION:
            return MeanAPCallable
        case TaskType.INSTANCE_SEGMENTATION:
            return MaskRLEMeanAPCallable
        case _:
            raise ValueError(f"Unsupported task type: {task.task_type}")


def get_otx_dataset_class_by_task_type(otx_task_type: OTXTaskType) -> type[OTXDataset]:
    """Get the OTXDataset class corresponding to the given OTXTaskType."""
    otx_task_type_to_class: dict[OTXTaskType, type[OTXDataset]] = {
        OTXTaskType.MULTI_CLASS_CLS: OTXMulticlassClsDataset,
        OTXTaskType.MULTI_LABEL_CLS: OTXMultilabelClsDataset,
        OTXTaskType.DETECTION: OTXDetectionDataset,
        OTXTaskType.INSTANCE_SEGMENTATION: OTXInstanceSegDataset,
    }
    try:
        return otx_task_type_to_class[otx_task_type]
    except KeyError:
        raise ValueError(f"Unsupported OTX task type: {otx_task_type}")


def convert_metrics(metrics: dict) -> dict[str, float]:
    """Convert metric values to a flat dict of scalar floats.

    Handles torch.Tensor values that may contain multiple elements
    (e.g., per-class metrics) by skipping them, matching the behavior
    of OVEngine.log_results.
    """
    result: dict[str, float] = {}
    for k, v in metrics.items():
        name = k.split("/")[1] if "/" in k else k
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                result[name] = v.item()
            else:
                logger.debug("Skipping non-scalar metric '{}' with {} elements", name, v.numel())
        else:
            result[name] = float(v)
    return result
