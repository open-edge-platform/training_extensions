# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for UltralyticsConfigurator and config dataclasses."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from getitune.backend.ultralytics.config import (
    UltralyticsConfig,
    UltralyticsTrainConfig,
)
from getitune.backend.ultralytics.configurator import (
    UltralyticsConfigurator,
    _deep_merge,
    _flatten_overrides,
)
from getitune.types.label import LabelInfo

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_RECIPE_DIR = Path(__file__).resolve().parents[4] / "src" / "getitune" / "recipe"

_DETECTION_CLASS_PATH = "getitune.backend.ultralytics.models.detection.UltralyticsDetectionModel"


def _minimal_recipe(
    *,
    backend: str = "ultralytics",
    task: str = "DETECTION",
    model_name: str = "yolo26n.pt",
) -> dict:
    """Return a minimal valid recipe dict."""
    return {
        "backend": backend,
        "task": task,
        "model": {
            "class_path": _DETECTION_CLASS_PATH,
            "init_args": {
                "model_name": model_name,
                "pretrained": True,
                "imgsz": 640,
            },
        },
        "engine": {"device": "auto"},
        "training": {
            "epochs": 50,
            "batch": 8,
            "lr0": 0.005,
            "close_mosaic": 0,
        },
        "export": {"format": "OPENVINO", "precision": "FP32"},
    }


def _write_recipe(tmp_path: Path, recipe: dict, name: str = "recipe.yaml") -> Path:
    """Write a recipe dict to a YAML file and return the path."""
    path = tmp_path / name
    path.write_text(yaml.dump(recipe), encoding="utf-8")
    return path


def _make_label_info(num_classes: int = 5) -> LabelInfo:
    return LabelInfo.from_num_classes(num_classes)


class TestUltralyticsTrainConfig:
    def test_to_train_args_returns_all_fields(self) -> None:
        cfg = UltralyticsTrainConfig(epochs=50, batch=8, lr0=0.005)
        args = cfg.to_train_args()
        assert args["epochs"] == 50
        assert args["batch"] == 8
        assert args["lr0"] == 0.005
        assert args["close_mosaic"] == 0  # default
        assert "optimizer" in args

    def test_defaults(self) -> None:
        cfg = UltralyticsTrainConfig()
        assert cfg.epochs == 100
        assert cfg.batch == 16
        assert cfg.patience == 100
        assert cfg.close_mosaic == 0


class TestUltralyticsConfig:
    def test_default_backend(self) -> None:
        cfg = UltralyticsConfig()
        assert cfg.backend == "ultralytics"

    def test_nested_defaults(self) -> None:
        cfg = UltralyticsConfig()
        assert cfg.model.pretrained is True
        assert cfg.engine.device == "auto"
        assert cfg.export.format == "OPENVINO"


class TestFromRecipe:
    def test_loads_valid_recipe(self, tmp_path: Path) -> None:
        path = _write_recipe(tmp_path, _minimal_recipe())
        cfg = UltralyticsConfigurator.from_recipe(path)
        assert cfg.config.backend == "ultralytics"
        assert cfg.config.task == "DETECTION"
        assert cfg.config.model.model_name == "yolo26n.pt"
        assert cfg.config.training.epochs == 50
        assert cfg.config.training.batch == 8

    def test_wrong_backend_raises(self, tmp_path: Path) -> None:
        recipe = _minimal_recipe(backend="lightning")
        path = _write_recipe(tmp_path, recipe)
        with pytest.raises(ValueError, match="Expected backend 'ultralytics'"):
            UltralyticsConfigurator.from_recipe(path)

    def test_unsupported_task_raises(self, tmp_path: Path) -> None:
        recipe = _minimal_recipe(task="SEMANTIC_SEGMENTATION")
        path = _write_recipe(tmp_path, recipe)
        with pytest.raises(ValueError, match="Unsupported task"):
            UltralyticsConfigurator.from_recipe(path)

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            UltralyticsConfigurator.from_recipe("/nonexistent/recipe.yaml")

    def test_non_mapping_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("- just a list\n", encoding="utf-8")
        with pytest.raises(TypeError, match="YAML mapping"):
            UltralyticsConfigurator.from_recipe(path)

    def test_resolves_data_reference(self, tmp_path: Path) -> None:
        # Write a base data YAML.
        base_dir = tmp_path / "_base_" / "data"
        base_dir.mkdir(parents=True)
        base_data = {"input_size": [640, 640], "train_subset": {"batch_size": 16}}
        (base_dir / "det.yaml").write_text(yaml.dump(base_data), encoding="utf-8")

        recipe = _minimal_recipe()
        recipe["data"] = "../_base_/data/det.yaml"
        recipe_dir = tmp_path / "detection"
        recipe_dir.mkdir()
        path = _write_recipe(recipe_dir, recipe)

        cfg = UltralyticsConfigurator.from_recipe(path)
        assert cfg.data_config["input_size"] == [640, 640]

    def test_inline_data(self, tmp_path: Path) -> None:
        recipe = _minimal_recipe()
        recipe["data"] = {"input_size": [320, 320]}
        path = _write_recipe(tmp_path, recipe)
        cfg = UltralyticsConfigurator.from_recipe(path)
        assert cfg.data_config["input_size"] == [320, 320]

    def test_data_overrides_merge(self, tmp_path: Path) -> None:
        # Base has batch_size=16, override sets it to 4.
        base_dir = tmp_path / "_base_" / "data"
        base_dir.mkdir(parents=True)
        base_data = {"train_subset": {"batch_size": 16, "num_workers": 8}}
        (base_dir / "det.yaml").write_text(yaml.dump(base_data), encoding="utf-8")

        recipe = _minimal_recipe()
        recipe["data"] = "../_base_/data/det.yaml"
        recipe["overrides"] = {"data": {"train_subset": {"batch_size": 4}}}
        recipe_dir = tmp_path / "detection"
        recipe_dir.mkdir()
        path = _write_recipe(recipe_dir, recipe)

        cfg = UltralyticsConfigurator.from_recipe(path)
        assert cfg.data_config["train_subset"]["batch_size"] == 4
        assert cfg.data_config["train_subset"]["num_workers"] == 8  # preserved

    def test_missing_data_ref_returns_empty(self, tmp_path: Path) -> None:
        recipe = _minimal_recipe()
        recipe["data"] = "nonexistent.yaml"
        path = _write_recipe(tmp_path, recipe)
        cfg = UltralyticsConfigurator.from_recipe(path)
        assert cfg.data_config == {}


class TestRealRecipes:
    """Verify the shipped YOLO26 recipe files parse correctly."""

    @pytest.mark.parametrize("variant", ["yolo26_n", "yolo26_s", "yolo26_m"])
    def test_yolo26_detection_recipe_loads(self, variant: str) -> None:
        path = _RECIPE_DIR / "detection" / f"{variant}.yaml"
        if not path.exists():
            pytest.skip(f"Recipe not found: {path}")
        cfg = UltralyticsConfigurator.from_recipe(path)
        assert cfg.config.backend == "ultralytics"
        assert cfg.config.task == "DETECTION"
        assert cfg.config.model.class_path == _DETECTION_CLASS_PATH
        assert cfg.config.model.pretrained is True
        assert cfg.config.training.close_mosaic == 0

    @pytest.mark.parametrize("variant", ["yolo26_n", "yolo26_s", "yolo26_m"])
    def test_yolo26_recipe_has_data_config(self, variant: str) -> None:
        path = _RECIPE_DIR / "detection" / f"{variant}.yaml"
        if not path.exists():
            pytest.skip(f"Recipe not found: {path}")
        cfg = UltralyticsConfigurator.from_recipe(path)
        assert "input_size" in cfg.data_config
        assert "train_subset" in cfg.data_config

    def test_new_variant_only_needs_recipe(self) -> None:
        """Adding a new compatible variant requires only a recipe change."""
        path = _RECIPE_DIR / "detection" / "yolo26_n.yaml"
        if not path.exists():
            pytest.skip("yolo26_n recipe not found")

        cfg = UltralyticsConfigurator.from_recipe(path)
        # Simulate switching to a different variant via override.
        cfg.apply_overrides({"model.model_name": "yolo26x.pt"})
        assert cfg.config.model.model_name == "yolo26x.pt"
        # Model creation still works with the same class.
        model = cfg.create_model(_make_label_info())
        assert model.model_name == "yolo26x.pt"


class TestApplyOverrides:
    def test_flat_overrides(self, tmp_path: Path) -> None:
        path = _write_recipe(tmp_path, _minimal_recipe())
        cfg = UltralyticsConfigurator.from_recipe(path)
        cfg.apply_overrides({"training.epochs": 200, "model.imgsz": 320})
        assert cfg.config.training.epochs == 200
        assert cfg.config.model.imgsz == 320

    def test_nested_overrides(self, tmp_path: Path) -> None:
        path = _write_recipe(tmp_path, _minimal_recipe())
        cfg = UltralyticsConfigurator.from_recipe(path)
        cfg.apply_overrides({"training": {"lr0": 0.001, "batch": 32}})
        assert cfg.config.training.lr0 == 0.001
        assert cfg.config.training.batch == 32

    def test_unknown_section_raises(self, tmp_path: Path) -> None:
        path = _write_recipe(tmp_path, _minimal_recipe())
        cfg = UltralyticsConfigurator.from_recipe(path)
        with pytest.raises(ValueError, match="Unknown override section"):
            cfg.apply_overrides({"bogus.field": 1})

    def test_unknown_field_raises(self, tmp_path: Path) -> None:
        path = _write_recipe(tmp_path, _minimal_recipe())
        cfg = UltralyticsConfigurator.from_recipe(path)
        with pytest.raises(ValueError, match="Unknown override"):
            cfg.apply_overrides({"training.nonexistent": 1})

    def test_bare_key_raises(self, tmp_path: Path) -> None:
        path = _write_recipe(tmp_path, _minimal_recipe())
        cfg = UltralyticsConfigurator.from_recipe(path)
        with pytest.raises(ValueError, match="section.field"):
            cfg.apply_overrides({"epochs": 10})

    def test_none_overrides_noop(self, tmp_path: Path) -> None:
        path = _write_recipe(tmp_path, _minimal_recipe())
        cfg = UltralyticsConfigurator.from_recipe(path)
        original_epochs = cfg.config.training.epochs
        cfg.apply_overrides(None)
        assert cfg.config.training.epochs == original_epochs


class TestApplyGetiOverrides:
    def _load(self, tmp_path: Path) -> UltralyticsConfigurator:
        recipe = _minimal_recipe()
        recipe["data"] = {"input_size": [640, 640], "train_subset": {"batch_size": 8}, "val_subset": {"batch_size": 8}}
        return UltralyticsConfigurator.from_recipe(_write_recipe(tmp_path, recipe))

    def test_learning_rate(self, tmp_path: Path) -> None:
        cfg = self._load(tmp_path)
        cfg.apply_geti_overrides({"training": {"learning_rate": 0.002}})
        assert cfg.config.training.lr0 == 0.002

    def test_batch_size_syncs_data(self, tmp_path: Path) -> None:
        cfg = self._load(tmp_path)
        cfg.apply_geti_overrides({"training": {"batch_size": 32}})
        assert cfg.config.training.batch == 32
        assert cfg.data_config["train_subset"]["batch_size"] == 32
        assert cfg.data_config["val_subset"]["batch_size"] == 32

    def test_max_epochs(self, tmp_path: Path) -> None:
        cfg = self._load(tmp_path)
        cfg.apply_geti_overrides({"training": {"max_epochs": 200}})
        assert cfg.config.training.epochs == 200

    def test_weight_decay(self, tmp_path: Path) -> None:
        cfg = self._load(tmp_path)
        cfg.apply_geti_overrides({"training": {"weight_decay": 0.001}})
        assert cfg.config.training.weight_decay == 0.001

    def test_input_size(self, tmp_path: Path) -> None:
        cfg = self._load(tmp_path)
        cfg.apply_geti_overrides({"training": {"input_size_height": 320, "input_size_width": 320}})
        assert cfg.config.model.imgsz == 320
        assert cfg.data_config["input_size"] == [320, 320]

    def test_early_stopping_patience(self, tmp_path: Path) -> None:
        cfg = self._load(tmp_path)
        cfg.apply_geti_overrides({"training": {"early_stopping": {"enable": True, "patience": 25}}})
        assert cfg.config.training.patience == 25

    def test_early_stopping_disable(self, tmp_path: Path) -> None:
        cfg = self._load(tmp_path)
        cfg.apply_geti_overrides({"training": {"early_stopping": {"enable": False}}})
        assert cfg.config.training.patience == 0

    def test_empty_overrides_noop(self, tmp_path: Path) -> None:
        cfg = self._load(tmp_path)
        original = cfg.config.training.lr0
        cfg.apply_geti_overrides({})
        assert cfg.config.training.lr0 == original


class TestToGetiConfig:
    def test_output_keys(self, tmp_path: Path) -> None:
        recipe = _minimal_recipe()
        recipe["data"] = {"input_size": [640, 640]}
        cfg = UltralyticsConfigurator.from_recipe(_write_recipe(tmp_path, recipe))
        geti = cfg.to_geti_config()

        assert geti["backend"] == "ultralytics"
        assert geti["task"] == "DETECTION"
        assert "model" in geti
        assert "data" in geti
        assert "max_epochs" in geti
        assert "training" in geti
        assert "export" in geti
        assert "callbacks" in geti
        assert geti["callbacks"] == []

    def test_model_shape(self, tmp_path: Path) -> None:
        cfg = UltralyticsConfigurator.from_recipe(_write_recipe(tmp_path, _minimal_recipe()))
        geti = cfg.to_geti_config()
        model = geti["model"]
        assert model["class_path"] == _DETECTION_CLASS_PATH
        assert model["init_args"]["model_name"] == "yolo26n.pt"

    def test_max_epochs_from_training(self, tmp_path: Path) -> None:
        cfg = UltralyticsConfigurator.from_recipe(_write_recipe(tmp_path, _minimal_recipe()))
        geti = cfg.to_geti_config()
        assert geti["max_epochs"] == 50  # from recipe training.epochs

    def test_data_is_deep_copy(self, tmp_path: Path) -> None:
        recipe = _minimal_recipe()
        recipe["data"] = {"input_size": [640, 640]}
        cfg = UltralyticsConfigurator.from_recipe(_write_recipe(tmp_path, recipe))
        geti = cfg.to_geti_config()
        # Mutating output should not affect configurator state.
        geti["data"]["input_size"] = [999, 999]
        assert cfg.data_config["input_size"] == [640, 640]

    def test_roundtrip_with_geti_overrides(self, tmp_path: Path) -> None:
        recipe = _minimal_recipe()
        recipe["data"] = {"input_size": [640, 640], "train_subset": {"batch_size": 8}, "val_subset": {"batch_size": 8}}
        cfg = UltralyticsConfigurator.from_recipe(_write_recipe(tmp_path, recipe))
        cfg.apply_geti_overrides({"training": {"learning_rate": 0.002, "batch_size": 4, "max_epochs": 10}})
        geti = cfg.to_geti_config()

        assert geti["max_epochs"] == 10
        assert geti["training"]["lr0"] == 0.002
        assert geti["training"]["batch"] == 4
        assert geti["data"]["train_subset"]["batch_size"] == 4


class TestCreateModel:
    def test_creates_detection_model(self, tmp_path: Path) -> None:
        cfg = UltralyticsConfigurator.from_recipe(_write_recipe(tmp_path, _minimal_recipe()))
        model = cfg.create_model(_make_label_info())
        assert model.model_name == "yolo26n.pt"
        assert model.label_info is not None
        assert model.label_info.num_classes == 5
        assert model.pretrained is True
        assert model.imgsz == 640

    def test_weights_path_overrides_model_name(self, tmp_path: Path) -> None:
        cfg = UltralyticsConfigurator.from_recipe(_write_recipe(tmp_path, _minimal_recipe()))
        model = cfg.create_model(_make_label_info(), weights_path="/custom/weights.pt")
        assert model.model_name == "/custom/weights.pt"

    def test_task_fallback_when_no_class_path(self, tmp_path: Path) -> None:
        recipe = _minimal_recipe()
        recipe["model"]["class_path"] = ""
        path = _write_recipe(tmp_path, recipe)
        cfg = UltralyticsConfigurator.from_recipe(path)
        model = cfg.create_model(_make_label_info())
        # Should resolve to UltralyticsDetectionModel via task lookup.
        from getitune.backend.ultralytics.models.detection import UltralyticsDetectionModel

        assert isinstance(model, UltralyticsDetectionModel)

    def test_invalid_class_path_raises(self, tmp_path: Path) -> None:
        recipe = _minimal_recipe()
        recipe["model"]["class_path"] = "nonexistent.module.BadClass"
        recipe["task"] = "DETECTION"
        path = _write_recipe(tmp_path, recipe)
        cfg = UltralyticsConfigurator.from_recipe(path)
        with pytest.raises(ValueError, match="Cannot import module"):
            cfg.create_model(_make_label_info())


class TestCreateEngine:
    def test_creates_engine(self, tmp_path: Path) -> None:
        cfg = UltralyticsConfigurator.from_recipe(_write_recipe(tmp_path, _minimal_recipe()))
        model = cfg.create_model(_make_label_info())
        engine = cfg.create_engine(model, data=tmp_path, work_dir=tmp_path / "work")
        from getitune.backend.ultralytics.engine import UltralyticsEngine

        assert isinstance(engine, UltralyticsEngine)

    def test_training_defaults_forwarded(self, tmp_path: Path) -> None:
        cfg = UltralyticsConfigurator.from_recipe(_write_recipe(tmp_path, _minimal_recipe()))
        model = cfg.create_model(_make_label_info())
        engine = cfg.create_engine(model, data=tmp_path, work_dir=tmp_path / "work")
        # Training defaults should be in engine._kwargs.
        assert engine._kwargs["epochs"] == 50
        assert engine._kwargs["close_mosaic"] == 0

    def test_engine_kwargs_override_recipe(self, tmp_path: Path) -> None:
        cfg = UltralyticsConfigurator.from_recipe(_write_recipe(tmp_path, _minimal_recipe()))
        model = cfg.create_model(_make_label_info())
        engine = cfg.create_engine(model, data=tmp_path, work_dir=tmp_path / "work", epochs=999)
        assert engine._kwargs["epochs"] == 999

    def test_device_override(self, tmp_path: Path) -> None:
        cfg = UltralyticsConfigurator.from_recipe(_write_recipe(tmp_path, _minimal_recipe()))
        model = cfg.create_model(_make_label_info())
        engine = cfg.create_engine(model, data=tmp_path, work_dir=tmp_path / "work", device="cpu")
        import torch

        assert engine._device == torch.device("cpu")


class TestFlattenOverrides:
    def test_flat(self) -> None:
        assert _flatten_overrides({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_nested(self) -> None:
        assert _flatten_overrides({"a": {"b": 1}}) == {"a.b": 1}

    def test_deeply_nested(self) -> None:
        assert _flatten_overrides({"a": {"b": {"c": 3}}}) == {"a.b.c": 3}


class TestDeepMerge:
    def test_simple(self) -> None:
        base = {"a": 1, "b": 2}
        _deep_merge(base, {"b": 3, "c": 4})
        assert base == {"a": 1, "b": 3, "c": 4}

    def test_nested(self) -> None:
        base = {"x": {"a": 1, "b": 2}}
        _deep_merge(base, {"x": {"b": 99}})
        assert base == {"x": {"a": 1, "b": 99}}

    def test_override_replaces_non_dict(self) -> None:
        base = {"x": [1, 2]}
        _deep_merge(base, {"x": [3]})
        assert base == {"x": [3]}
