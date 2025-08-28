# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTX Dice metric used for the OTX semantic segmentation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchmetrics.metric import Metric

from otx.types.label import AnomalyLabelInfo

if TYPE_CHECKING:
    from torch import Tensor

from anomalib.metrics import AnomalibMetricCollection, create_metric_collection


class OTXAnomalyMetric(Metric):
    """Wrapper for Anomalib metrics for OTX anomaly task."""

    def __init__(self, label_info: AnomalyLabelInfo) -> None:
        super().__init__()
        self.label_info = label_info
        metric_names = ["AUROC", "F1Score"]
        self.image_metrics: AnomalibMetricCollection = create_metric_collection(metric_names, prefix="image_")
        self.pixel_metrics: AnomalibMetricCollection = create_metric_collection(metric_names, prefix="pixel_")
        self.reset()

    def update(
        self,
        pred_scores: Tensor,
        labels: Tensor,
        anomaly_maps: Tensor | None = None,
        masks: Tensor | None = None,
    ) -> None:
        """Update performance metrics."""
        self.image_metrics.update(pred_scores, labels.int())
        if masks is not None and anomaly_maps is not None:
            self.pixel_metrics.update(anomaly_maps, masks.int())

    def reset(self) -> None:
        """Reset for every validation and test epoch.

        Please be careful that some variables should not be reset for each epoch.
        """
        self.image_metrics.reset()
        self.pixel_metrics.reset()

    def compute(self) -> dict[str, float]:
        """Compute the metrics."""
        image_results = self.image_metrics.compute()
        pixel_results = self.pixel_metrics.compute()

        return {**image_results, **pixel_results}


def _anomaly_metrics_callable(label_info: AnomalyLabelInfo = AnomalyLabelInfo()) -> OTXAnomalyMetric:
    return OTXAnomalyMetric(label_info=label_info)


AnomalyCallable = _anomaly_metrics_callable
