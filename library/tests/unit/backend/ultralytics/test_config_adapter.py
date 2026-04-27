# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for UltralyticsConfigAdapter."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from getitune.backend.ultralytics.config_adapter import UltralyticsConfigAdapter
from getitune.backend.ultralytics.configurator import UltralyticsConfigurator

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_RECIPE_DIR = Path(__file__).resolve().parents[4] / "src" / "getitune" / "recipe"


def _minimal_ultralytics_recipe() -> dict:
    """Return a minimal valid Ultralytics recipe dict."""
    return {
        "backend": "ultralytics",
        "task": "DETECTION",
        "model": {
            "class_path": "getitune.backend.ultralytics.models.detection.UltralyticsDetectionModel",
            "init_args": {
                "model_name": "yolo26n.yaml",
                "pretrained": False,
                "imgsz": 640,
            },
        },
        "engine": {"device": "auto"},
        "training": {
            "epochs": 100,
            "batch": 16,
            "lr0": 0.01,
            "weight_decay": 0.0005,
            "patience": 100,
            "close_mosaic": 0,
        },
        "export": {"format": "OPENVINO", "precision": "FP32"},
        "data": {
            "input_size": [640, 640],
            "train_subset": {"batch_size": 16, "augmentations_cpu": []},
            "val_subset": {"batch_size": 16},
            "test_subset": {"batch_size": 16},
        },
    }


def _write_recipe(tmp_path: Path, recipe: dict, name: str = "recipe.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.dump(recipe), encoding="utf-8")
    return path


# ------------------------------------------------------------------
# is_ultralytics_recipe
# ------------------------------------------------------------------


class TestIsUltralyticsRecipe:
    def test_true_for_ultralytics_backend(self, tmp_path: Path) -> None:
        path = _write_recipe(tmp_path, {"backend": "ultralytics", "task": "DETECTION"})
        assert UltralyticsConfigAdapter.is_ultralytics_recipe(path) is True

    def test_false_for_lightning_backend(self, tmp_path: Path) -> None:
        path = _write_recipe(tmp_path, {"backend": "lightning", "task": "DETECTION"})
        assert UltralyticsConfigAdapter.is_ultralytics_recipe(path) is False

    def test_false_for_no_backend(self, tmp_path: Path) -> None:
        path = _write_recipe(tmp_path, {"task": "DETECTION"})
        assert UltralyticsConfigAdapter.is_ultralytics_recipe(path) is False


# ------------------------------------------------------------------
# convert (full round-trip)
# ------------------------------------------------------------------


class TestConvert:
    def test_convert_produces_backend_tagged_dict(self, tmp_path: Path) -> None:
        path = _write_recipe(tmp_path, _minimal_ultralytics_recipe())
        result = UltralyticsConfigAdapter.convert(path)
        assert result["backend"] == "ultralytics"
        assert result["task"] == "DETECTION"
        assert result["model"]["init_args"]["model_name"] == "yolo26n.yaml"

    def test_convert_applies_hyper_parameters(self, tmp_path: Path) -> None:
        path = _write_recipe(tmp_path, _minimal_ultralytics_recipe())
        result = UltralyticsConfigAdapter.convert(
            path,
            hyper_parameters={"training": {"learning_rate": 0.002, "batch_size": 4, "max_epochs": 10}},
        )
        assert result["training"]["lr0"] == 0.002
        assert result["training"]["batch"] == 4
        assert result["training"]["epochs"] == 10
        assert result["data"]["train_subset"]["batch_size"] == 4

    def test_convert_without_hyper_parameters(self, tmp_path: Path) -> None:
        path = _write_recipe(tmp_path, _minimal_ultralytics_recipe())
        result = UltralyticsConfigAdapter.convert(path, hyper_parameters=None)
        assert result["training"]["lr0"] == 0.01
        assert result["training"]["epochs"] == 100

    def test_convert_roundtrips_through_from_config_dict(self, tmp_path: Path) -> None:
        """Config dict from convert() must be consumable by from_config_dict()."""
        path = _write_recipe(tmp_path, _minimal_ultralytics_recipe())
        config_dict = UltralyticsConfigAdapter.convert(path)
        configurator = UltralyticsConfigurator.from_config_dict(config_dict)
        assert configurator.config.model.model_name == "yolo26n.yaml"
        assert configurator.config.training.epochs == 100


# ------------------------------------------------------------------
# apply_hyper_parameters
# ------------------------------------------------------------------


class TestApplyHyperParameters:
    def _make_configurator(self, tmp_path: Path) -> UltralyticsConfigurator:
        path = _write_recipe(tmp_path, _minimal_ultralytics_recipe())
        return UltralyticsConfigurator.from_recipe(path)

    def test_learning_rate(self, tmp_path: Path) -> None:
        cfg = self._make_configurator(tmp_path)
        UltralyticsConfigAdapter.apply_hyper_parameters(cfg, {"training": {"learning_rate": 0.05}})
        assert cfg.config.training.lr0 == 0.05

    def test_weight_decay(self, tmp_path: Path) -> None:
        cfg = self._make_configurator(tmp_path)
        UltralyticsConfigAdapter.apply_hyper_parameters(cfg, {"training": {"weight_decay": 0.001}})
        assert cfg.config.training.weight_decay == 0.001

    def test_max_epochs(self, tmp_path: Path) -> None:
        cfg = self._make_configurator(tmp_path)
        UltralyticsConfigAdapter.apply_hyper_parameters(cfg, {"training": {"max_epochs": 50}})
        assert cfg.config.training.epochs == 50

    def test_batch_size_propagates_to_data(self, tmp_path: Path) -> None:
        cfg = self._make_configurator(tmp_path)
        UltralyticsConfigAdapter.apply_hyper_parameters(cfg, {"training": {"batch_size": 8}})
        assert cfg.config.training.batch == 8
        assert cfg.data_config["train_subset"]["batch_size"] == 8
        assert cfg.data_config["val_subset"]["batch_size"] == 8
        assert cfg.data_config["test_subset"]["batch_size"] == 8

    def test_input_size(self, tmp_path: Path) -> None:
        cfg = self._make_configurator(tmp_path)
        UltralyticsConfigAdapter.apply_hyper_parameters(
            cfg, {"training": {"input_size_height": 320, "input_size_width": 320}}
        )
        assert cfg.config.model.imgsz == 320
        assert cfg.data_config["input_size"] == [320, 320]

    def test_early_stopping_disable(self, tmp_path: Path) -> None:
        cfg = self._make_configurator(tmp_path)
        UltralyticsConfigAdapter.apply_hyper_parameters(cfg, {"training": {"early_stopping": {"enable": False}}})
        assert cfg.config.training.patience == 0

    def test_early_stopping_set_patience(self, tmp_path: Path) -> None:
        cfg = self._make_configurator(tmp_path)
        UltralyticsConfigAdapter.apply_hyper_parameters(
            cfg, {"training": {"early_stopping": {"enable": True, "patience": 25}}}
        )
        assert cfg.config.training.patience == 25


# ------------------------------------------------------------------
# Augmentations
# ------------------------------------------------------------------


class TestAugmentations:
    def _make_configurator(self, tmp_path: Path) -> UltralyticsConfigurator:
        path = _write_recipe(tmp_path, _minimal_ultralytics_recipe())
        return UltralyticsConfigurator.from_recipe(path)

    def test_tiling_enable(self, tmp_path: Path) -> None:
        cfg = self._make_configurator(tmp_path)
        UltralyticsConfigAdapter.apply_hyper_parameters(
            cfg,
            {
                "dataset_preparation": {
                    "augmentation": {
                        "tiling": {
                            "enable": True,
                            "enable_adaptive_tiling": True,
                            "tile_size": 512,
                            "tile_overlap": 0.3,
                        }
                    }
                }
            },
        )
        tc = cfg.data_config["tile_config"]
        assert tc["enable_tiler"] is True
        assert tc["enable_adaptive_tiling"] is True
        assert tc["tile_size"] == [512, 512]
        assert tc["overlap"] == 0.3

    def test_horizontal_flip_add(self, tmp_path: Path) -> None:
        cfg = self._make_configurator(tmp_path)
        UltralyticsConfigAdapter.apply_hyper_parameters(
            cfg,
            {"dataset_preparation": {"augmentation": {"random_horizontal_flip": {"enable": True, "probability": 0.7}}}},
        )
        aug_list = cfg.data_config["train_subset"]["augmentations_cpu"]
        hflip = [a for a in aug_list if "HorizontalFlip" in a["class_path"]]
        assert len(hflip) == 1
        assert hflip[0]["init_args"]["p"] == 0.7

    def test_horizontal_flip_remove(self, tmp_path: Path) -> None:
        recipe = _minimal_ultralytics_recipe()
        recipe["data"]["train_subset"]["augmentations_cpu"] = [
            {"class_path": "torchvision.transforms.v2.RandomHorizontalFlip", "init_args": {"p": 0.5}},
        ]
        cfg = UltralyticsConfigurator.from_config_dict(recipe)
        UltralyticsConfigAdapter.apply_hyper_parameters(
            cfg,
            {"dataset_preparation": {"augmentation": {"random_horizontal_flip": {"enable": False}}}},
        )
        aug_list = cfg.data_config["train_subset"]["augmentations_cpu"]
        hflip = [a for a in aug_list if "HorizontalFlip" in a["class_path"]]
        assert len(hflip) == 0

    def test_iou_random_crop_add(self, tmp_path: Path) -> None:
        cfg = self._make_configurator(tmp_path)
        UltralyticsConfigAdapter.apply_hyper_parameters(
            cfg,
            {"dataset_preparation": {"augmentation": {"iou_random_crop": {"enable": True}}}},
        )
        aug_list = cfg.data_config["train_subset"]["augmentations_cpu"]
        crop = [a for a in aug_list if "RandomIoUCrop" in a["class_path"]]
        assert len(crop) == 1


# ------------------------------------------------------------------
# to_config_dict
# ------------------------------------------------------------------


class TestToConfigDict:
    def test_keys_present(self, tmp_path: Path) -> None:
        cfg = UltralyticsConfigurator.from_config_dict(_minimal_ultralytics_recipe())
        result = UltralyticsConfigAdapter.to_config_dict(cfg)
        assert set(result.keys()) == {"backend", "task", "model", "engine", "training", "export", "data"}

    def test_model_section(self, tmp_path: Path) -> None:
        cfg = UltralyticsConfigurator.from_config_dict(_minimal_ultralytics_recipe())
        result = UltralyticsConfigAdapter.to_config_dict(cfg)
        assert result["model"]["init_args"]["model_name"] == "yolo26n.yaml"
        assert result["model"]["init_args"]["pretrained"] is False

    def test_data_is_deep_copy(self, tmp_path: Path) -> None:
        cfg = UltralyticsConfigurator.from_config_dict(_minimal_ultralytics_recipe())
        result = UltralyticsConfigAdapter.to_config_dict(cfg)
        # Mutating the output should not affect the configurator
        result["data"]["input_size"] = [999, 999]
        assert cfg.data_config["input_size"] == [640, 640]


# ------------------------------------------------------------------
# Real recipes
# ------------------------------------------------------------------


class TestRealRecipes:
    @pytest.mark.parametrize("variant", ["yolo26_n", "yolo26_s", "yolo26_m"])
    def test_convert_real_recipe(self, variant: str) -> None:
        path = _RECIPE_DIR / "detection" / f"{variant}.yaml"
        if not path.exists():
            pytest.skip(f"Recipe not found: {path}")
        result = UltralyticsConfigAdapter.convert(path)
        assert result["backend"] == "ultralytics"
        assert result["task"] == "DETECTION"
        assert result["model"]["init_args"]["model_name"].endswith(".yaml")
        assert result["model"]["init_args"]["pretrained"] is False

    @pytest.mark.parametrize("variant", ["yolo26_n", "yolo26_s", "yolo26_m"])
    def test_real_recipe_roundtrips(self, variant: str) -> None:
        """Config from convert() can reconstruct a configurator."""
        path = _RECIPE_DIR / "detection" / f"{variant}.yaml"
        if not path.exists():
            pytest.skip(f"Recipe not found: {path}")
        config_dict = UltralyticsConfigAdapter.convert(path)
        configurator = UltralyticsConfigurator.from_config_dict(config_dict)
        assert configurator.config.backend == "ultralytics"
