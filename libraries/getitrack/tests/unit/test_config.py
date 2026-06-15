# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Pydantic configuration models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from getitrack.config import (
    AlgorithmType,
    AssociationConfig,
    InterpolationMethod,
    LifecycleConfig,
    TrackerConfig,
)


class TestDefaults:
    def test_top_level_defaults(self):
        c = TrackerConfig()
        assert c.algorithm == AlgorithmType.BYTETRACK
        assert c.interpolation.method == InterpolationMethod.LINEAR

    def test_subconfig_defaults_match_plan(self):
        c = TrackerConfig()
        assert c.association.match_threshold == pytest.approx(0.8)
        assert c.association.class_filter is None
        assert c.lifecycle.min_hits == 2
        assert c.lifecycle.tentative_max_age == 0
        assert c.lifecycle.max_age == 30
        assert c.motion.velocity_decay == pytest.approx(0.99)
        assert c.appearance.enabled is False
        assert c.interpolation.enabled is True


class TestValidation:
    def test_match_threshold_out_of_range_raises(self):
        with pytest.raises(ValidationError, match="match_threshold"):
            AssociationConfig(match_threshold=2.0)

    def test_negative_max_age_raises(self):
        with pytest.raises(ValidationError, match="max_age"):
            LifecycleConfig(max_age=0)

    def test_unknown_field_raises(self):
        with pytest.raises(ValidationError, match="bogus"):
            TrackerConfig.model_validate({"algorithm": "bytetrack", "bogus": 1})

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValidationError):
            TrackerConfig.model_validate({"algorithm": "no-such-algo"})


class TestYAMLRoundTrip:
    def test_round_trip_lossless(self, tmp_path: Path):
        cfg = TrackerConfig(
            algorithm=AlgorithmType.OCSORT,
            lifecycle=LifecycleConfig(max_age=42),
        )
        p = tmp_path / "cfg.yaml"
        cfg.to_yaml(p)
        loaded = TrackerConfig.from_yaml(p)
        assert loaded == cfg

    def test_from_empty_yaml_returns_defaults(self, tmp_path: Path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        loaded = TrackerConfig.from_yaml(p)
        assert loaded == TrackerConfig()

    def test_default_config_loads(self):
        p = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
        loaded = TrackerConfig.from_yaml(p)
        assert loaded == TrackerConfig()


class TestModelValidate:
    def test_from_dict(self):
        cfg = TrackerConfig.model_validate(
            {
                "algorithm": "ocsort",
                "association": {"match_threshold": 0.4},
                "lifecycle": {"max_age": 50},
            },
        )
        assert cfg.algorithm == AlgorithmType.OCSORT
        assert cfg.association.match_threshold == pytest.approx(0.4)
        assert cfg.lifecycle.max_age == 50

    def test_json_schema_export_contains_subconfigs(self):
        schema = TrackerConfig.model_json_schema()
        names = set(schema.get("$defs", {}).keys())
        assert {"AssociationConfig", "LifecycleConfig", "MotionConfig"} <= names

    def test_json_schema_includes_field_descriptions(self):
        """Pydantic should harvest the attribute docstrings as field descriptions."""
        schema = TrackerConfig.model_json_schema()
        assoc = schema["$defs"]["AssociationConfig"]["properties"]
        assert "match_threshold" in assoc
        assert "description" in assoc["match_threshold"]
        assert "IoU" in assoc["match_threshold"]["description"]
