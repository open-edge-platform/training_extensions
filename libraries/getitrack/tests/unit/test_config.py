# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared Pydantic configuration models."""

import pytest
import yaml
from pydantic import ValidationError

from getitrack.config import (
    AlgorithmType,
    InterpolationConfig,
    InterpolationMethod,
    LifecycleConfig,
    MotionConfig,
    TrackerConfig,
)


class _DemoConfig(TrackerConfig):
    """Minimal concrete config for exercising the shared base."""

    algorithm: AlgorithmType = AlgorithmType.BYTETRACK


class TestSubConfigDefaults:
    def test_lifecycle_defaults(self):
        lc = LifecycleConfig()
        assert (lc.min_hits, lc.tentative_max_age, lc.max_age) == (2, 0, 30)

    def test_motion_default(self):
        assert MotionConfig().velocity_decay == pytest.approx(0.99)

    def test_interpolation_defaults(self):
        ic = InterpolationConfig()
        assert ic.enabled is True
        assert ic.method == InterpolationMethod.LINEAR


class TestBaseConfig:
    def test_shared_defaults(self):
        c = _DemoConfig()
        assert c.algorithm == AlgorithmType.BYTETRACK
        assert c.class_filter is None
        assert c.score_threshold == pytest.approx(0.1)
        assert c.lifecycle.min_hits == 2

    def test_unknown_field_raises(self):
        with pytest.raises(ValidationError, match="bogus"):
            _DemoConfig.model_validate({"bogus": 1})

    def test_negative_max_age_raises(self):
        with pytest.raises(ValidationError, match="max_age"):
            LifecycleConfig(max_age=0)

    def test_score_threshold_out_of_range_raises(self):
        with pytest.raises(ValidationError, match="score_threshold"):
            _DemoConfig(score_threshold=2.0)

    def test_variant_rejects_foreign_algorithm(self):
        # A subclass pins ``algorithm``; a different value must be rejected.
        with pytest.raises(ValidationError, match="pins algorithm"):
            _DemoConfig(algorithm=AlgorithmType.OCSORT)


class TestYAML:
    def test_to_yaml_serialises_fields(self, tmp_path):
        cfg = _DemoConfig(lifecycle=LifecycleConfig(max_age=42))
        p = tmp_path / "c.yaml"
        cfg.to_yaml(p)
        data = yaml.safe_load(p.read_text())
        assert data["algorithm"] == "bytetrack"
        assert data["lifecycle"]["max_age"] == 42

    def test_json_schema_exports_subconfigs_and_descriptions(self):
        schema = _DemoConfig.model_json_schema()
        names = set(schema.get("$defs", {}).keys())
        assert {"LifecycleConfig", "MotionConfig", "InterpolationConfig"} <= names
        assert "description" in schema["properties"]["score_threshold"]
