# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from app.api.schemas.evaluation import EvaluationMetricName, EvaluationView


class TestEvaluationViewPrimaryMetric:
    """Tests for the primary metric selection logic in EvaluationView."""

    def test_primary_metric_is_map50_for_detection_metrics(self):
        """When map_50 is present, mAP@0.5 should be the primary metric (not mAP@0.5:0.95)."""
        metrics = {
            "map": 0.85,
            "map_50": 0.90,
            "map_75": 0.80,
            "mar_1": 0.88,
            "mar_10": 0.90,
            "mar_100": 0.92,
        }

        result = EvaluationView._EvaluationView__serialize_metrics(metrics)

        primary_metrics = [m for m in result if m.primary]
        assert len(primary_metrics) == 1
        assert primary_metrics[0].name == EvaluationMetricName.MAP_50

    def test_primary_metric_is_map_when_map50_absent(self):
        """When only map is present (without map_50), mAP should remain the primary metric."""
        metrics = {
            "map": 0.85,
            "mar_1": 0.88,
        }

        result = EvaluationView._EvaluationView__serialize_metrics(metrics)

        primary_metrics = [m for m in result if m.primary]
        assert len(primary_metrics) == 1
        assert primary_metrics[0].name == EvaluationMetricName.MAP

    def test_primary_metric_is_accuracy_for_classification_metrics(self):
        """For classification tasks (no map key), accuracy should be the primary metric."""
        metrics = {
            "accuracy": 0.95,
            "precision": 0.96,
            "recall": 0.94,
            "f_measure": 0.95,
        }

        result = EvaluationView._EvaluationView__serialize_metrics(metrics)

        primary_metrics = [m for m in result if m.primary]
        assert len(primary_metrics) == 1
        assert primary_metrics[0].name == EvaluationMetricName.ACCURACY

    def test_map_is_not_primary_when_map50_present(self):
        """When map_50 is present, mAP@0.5:0.95 should NOT be the primary metric."""
        metrics = {
            "map": 0.85,
            "map_50": 0.90,
        }

        result = EvaluationView._EvaluationView__serialize_metrics(metrics)

        map_metric = next((m for m in result if m.name == EvaluationMetricName.MAP), None)
        assert map_metric is not None
        assert not map_metric.primary

    def test_raises_when_no_known_primary_metric(self):
        """Should raise ValueError when no known primary metric is found."""
        metrics = {
            "unknown_metric": 0.5,
        }

        with pytest.raises(ValueError, match="Unable to determine the primary evaluation metric"):
            EvaluationView._EvaluationView__serialize_metrics(metrics)
