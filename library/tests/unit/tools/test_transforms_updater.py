# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TransformsUpdater and HyperparametersUpdater classes."""

from __future__ import annotations

import pytest

from otx.tools.converter import HyperparametersUpdater, TransformsUpdater


class TestTransformsUpdater:
    """Test TransformsUpdater augmentation mapping and parameter remapping."""

    @pytest.fixture
    def base_config(self):
        """Create a base config with augmentations_cpu and augmentations_gpu."""
        return {
            "data": {
                "train_subset": {
                    "augmentations_cpu": [
                        {
                            "class_path": "otx.data.augmentation.transforms.Resize",
                            "init_args": {"size": "$(input_size)"},
                        }
                    ],
                    "augmentations_gpu": [
                        {
                            "class_path": "kornia.augmentation.Normalize",
                            "init_args": {
                                "mean": [0.485, 0.456, 0.406],
                                "std": [0.229, 0.224, 0.225],
                            },
                        }
                    ],
                },
                "tile_config": {"enable_tiler": False},
            }
        }

    def test_param_rename_simple(self, base_config):
        """Test simple parameter renames: probability -> p, sigma -> std."""
        aug_params = {
            "random_horizontal_flip": {
                "enable": True,
                "probability": 0.7,
            }
        }
        TransformsUpdater.update(aug_params, base_config)

        gpu_augs = base_config["data"]["train_subset"]["augmentations_gpu"]
        flip_aug = next(
            (a for a in gpu_augs if "RandomHorizontalFlip" in a.get("class_path", "")),
            None,
        )
        assert flip_aug is not None
        assert flip_aug["init_args"]["p"] == 0.7

    def test_param_rename_affine_translate(self, base_config):
        """Test affine parameter transform: max_translate_ratio -> translate [v, v]."""
        aug_params = {
            "random_affine": {
                "enable": True,
                "max_rotate_degree": 30.0,
                "max_translate_ratio": 0.15,
                "scaling_ratio_range": [0.8, 1.2],
            }
        }
        TransformsUpdater.update(aug_params, base_config)

        gpu_augs = base_config["data"]["train_subset"]["augmentations_gpu"]
        affine_aug = next(
            (a for a in gpu_augs if "RandomAffine" in a.get("class_path", "")),
            None,
        )
        assert affine_aug is not None
        assert affine_aug["init_args"]["degrees"] == 30.0
        assert affine_aug["init_args"]["translate"] == [0.15, 0.15]
        assert affine_aug["init_args"]["scale"] == [0.8, 1.2]

    def test_param_rename_affine_shear(self, base_config):
        """Test affine parameter transform: max_shear_degree -> shear [-v, v]."""
        aug_params = {
            "random_affine": {
                "enable": True,
                "max_shear_degree": 5.0,
            }
        }
        TransformsUpdater.update(aug_params, base_config)

        gpu_augs = base_config["data"]["train_subset"]["augmentations_gpu"]
        affine_aug = next(
            (a for a in gpu_augs if "RandomAffine" in a.get("class_path", "")),
            None,
        )
        assert affine_aug is not None
        assert affine_aug["init_args"]["shear"] == [-5.0, 5.0]

    def test_param_value_already_list(self, base_config):
        """Test that list values pass through unchanged."""
        aug_params = {
            "color_jitter": {
                "enable": True,
                "brightness": [0.875, 1.125],
                "contrast": [0.5, 1.5],
            }
        }
        TransformsUpdater.update(aug_params, base_config)

        gpu_augs = base_config["data"]["train_subset"]["augmentations_gpu"]
        jitter_aug = next(
            (a for a in gpu_augs if "ColorJiggle" in a.get("class_path", "")),
            None,
        )
        assert jitter_aug is not None
        assert jitter_aug["init_args"]["brightness"] == [0.875, 1.125]
        assert jitter_aug["init_args"]["contrast"] == [0.5, 1.5]

    def test_kernel_size_scalar_to_list(self, base_config):
        """Test kernel_size transforms int scalar to [v, v]."""
        aug_params = {
            "gaussian_blur": {
                "enable": True,
                "kernel_size": 5,
            }
        }
        TransformsUpdater.update(aug_params, base_config)

        gpu_augs = base_config["data"]["train_subset"]["augmentations_gpu"]
        blur_aug = next(
            (a for a in gpu_augs if "RandomGaussianBlur" in a.get("class_path", "")),
            None,
        )
        assert blur_aug is not None
        assert blur_aug["init_args"]["kernel_size"] == [5, 5]

    def test_add_new_augmentation_cpu(self, base_config):
        """Test adding new CPU augmentation when not present in config."""
        aug_params = {
            "random_resize_crop": {
                "enable": True,
                "scale": [0.1, 1.0],
            }
        }
        TransformsUpdater.update(aug_params, base_config)

        cpu_augs = base_config["data"]["train_subset"]["augmentations_cpu"]
        crop_aug = next(
            (a for a in cpu_augs if "RandomResizedCrop" in a.get("class_path", "")),
            None,
        )
        assert crop_aug is not None
        assert crop_aug["init_args"]["scale"] == [0.1, 1.0]

    def test_add_iou_random_crop_with_probability(self, base_config):
        """Test adding OTX RandomIoUCrop and mapping probability to p."""
        # Remove existing RandomIoUCrop from base config first to force insertion path
        base_config["data"]["train_subset"]["augmentations_cpu"] = [
            aug
            for aug in base_config["data"]["train_subset"]["augmentations_cpu"]
            if "RandomIoUCrop" not in aug.get("class_path", "")
        ]

        aug_params = {
            "iou_random_crop": {
                "enable": True,
                "probability": 0.8,
                "trials": 60,
            }
        }
        TransformsUpdater.update(aug_params, base_config)

        cpu_augs = base_config["data"]["train_subset"]["augmentations_cpu"]
        iou_aug = next(
            (a for a in cpu_augs if "RandomIoUCrop" in a.get("class_path", "")),
            None,
        )
        assert iou_aug is not None
        assert iou_aug["class_path"] == "otx.data.augmentation.transforms.RandomIoUCrop"
        assert iou_aug["init_args"]["p"] == 0.8
        assert iou_aug["init_args"]["trials"] == 60

    def test_add_new_augmentation_gpu(self, base_config):
        """Test adding new GPU augmentation when not present in config."""
        aug_params = {
            "gaussian_noise": {
                "enable": True,
                "mean": 0.0,
                "sigma": 0.05,
            }
        }
        TransformsUpdater.update(aug_params, base_config)

        gpu_augs = base_config["data"]["train_subset"]["augmentations_gpu"]
        noise_aug = next(
            (a for a in gpu_augs if "RandomGaussianNoise" in a.get("class_path", "")),
            None,
        )
        assert noise_aug is not None
        assert noise_aug["init_args"]["mean"] == 0.0
        assert noise_aug["init_args"]["std"] == 0.05  # sigma renamed to std

    def test_update_existing_augmentation(self, base_config):
        """Test updating parameters of existing augmentation."""
        # Add flip to config first
        base_config["data"]["train_subset"]["augmentations_gpu"].insert(
            0,
            {
                "class_path": "kornia.augmentation.RandomHorizontalFlip",
                "init_args": {"p": 0.5},
            },
        )

        aug_params = {
            "random_horizontal_flip": {
                "enable": True,
                "probability": 0.9,
            }
        }
        TransformsUpdater.update(aug_params, base_config)

        gpu_augs = base_config["data"]["train_subset"]["augmentations_gpu"]
        flip_aug = next(
            (a for a in gpu_augs if "RandomHorizontalFlip" in a.get("class_path", "")),
            None,
        )
        assert flip_aug is not None
        assert flip_aug["init_args"]["p"] == 0.9

    def test_disable_augmentation(self, base_config):
        """Test disabling (removing) an existing augmentation."""
        base_config["data"]["train_subset"]["augmentations_gpu"].insert(
            0,
            {
                "class_path": "kornia.augmentation.RandomVerticalFlip",
                "init_args": {"p": 0.5},
            },
        )

        aug_params = {
            "random_vertical_flip": {
                "enable": False,
            }
        }
        TransformsUpdater.update(aug_params, base_config)

        gpu_augs = base_config["data"]["train_subset"]["augmentations_gpu"]
        vflip_aug = next(
            (a for a in gpu_augs if "RandomVerticalFlip" in a.get("class_path", "")),
            None,
        )
        assert vflip_aug is None

    def test_disable_random_resize_crop_replaces_with_resize(self, base_config):
        """Test that disabling random_resize_crop replaces it with Resize."""
        base_config["data"]["train_subset"]["augmentations_cpu"].insert(
            0,
            {
                "class_path": "torchvision.transforms.v2.RandomResizedCrop",
                "init_args": {"size": "$(input_size)"},
            },
        )

        aug_params = {
            "random_resize_crop": {
                "enable": False,
            }
        }
        TransformsUpdater.update(aug_params, base_config)

        cpu_augs = base_config["data"]["train_subset"]["augmentations_cpu"]
        resize_aug = next(
            (a for a in cpu_augs if a["class_path"] == "otx.data.augmentation.transforms.Resize"),
            None,
        )
        assert resize_aug is not None
        assert "RandomResizedCrop" not in [a.get("class_path", "") for a in cpu_augs]

    def test_insert_position_gpu_before_normalize(self, base_config):
        """Test that new GPU augmentation is inserted before Normalize."""
        aug_params = {
            "color_jitter": {
                "enable": True,
                "brightness": 0.2,
            }
        }
        TransformsUpdater.update(aug_params, base_config)

        gpu_augs = base_config["data"]["train_subset"]["augmentations_gpu"]
        normalize_idx = next(
            (i for i, a in enumerate(gpu_augs) if "Normalize" in a.get("class_path", "")),
            None,
        )
        jitter_idx = next(
            (i for i, a in enumerate(gpu_augs) if "ColorJiggle" in a.get("class_path", "")),
            None,
        )
        assert jitter_idx is not None
        assert normalize_idx is not None
        assert jitter_idx < normalize_idx

    def test_insert_position_cpu_before_resize(self, base_config):
        """Test that new CPU augmentation is inserted before Resize."""
        aug_params = {
            "iou_random_crop": {
                "enable": True,
            }
        }
        TransformsUpdater.update(aug_params, base_config)

        cpu_augs = base_config["data"]["train_subset"]["augmentations_cpu"]
        resize_idx = next(
            (
                i
                for i, a in enumerate(cpu_augs)
                if "Resize" in a.get("class_path", "") and "RandomResizedCrop" not in a.get("class_path", "")
            ),
            None,
        )
        crop_idx = next(
            (
                i
                for i, a in enumerate(cpu_augs)
                if "MinIoURandomCrop" in a.get("class_path", "") or "RandomIoUCrop" in a.get("class_path", "")
            ),
            None,
        )
        assert crop_idx is not None
        assert resize_idx is not None
        assert crop_idx < resize_idx

    def test_unknown_augmentation_raises_error(self, base_config):
        """Test that unknown augmentation name raises ValueError."""
        aug_params = {
            "unknown_aug": {
                "enable": True,
            }
        }
        with pytest.raises(ValueError, match="Unknown augmentation"):
            TransformsUpdater.update(aug_params, base_config)

    def test_empty_augmentation_params(self, base_config):
        """Test that empty augmentation params don't modify config."""
        original_gpu_len = len(base_config["data"]["train_subset"]["augmentations_gpu"])
        original_cpu_len = len(base_config["data"]["train_subset"]["augmentations_cpu"])

        aug_params = {}
        TransformsUpdater.update(aug_params, base_config)

        assert len(base_config["data"]["train_subset"]["augmentations_gpu"]) == original_gpu_len
        assert len(base_config["data"]["train_subset"]["augmentations_cpu"]) == original_cpu_len

    def test_none_param_values_are_skipped(self, base_config):
        """Test that None parameter values are skipped."""
        aug_params = {
            "color_jitter": {
                "enable": True,
                "brightness": 0.2,
                "contrast": None,  # Should be skipped
            }
        }
        TransformsUpdater.update(aug_params, base_config)

        gpu_augs = base_config["data"]["train_subset"]["augmentations_gpu"]
        jitter_aug = next(
            (a for a in gpu_augs if "ColorJiggle" in a.get("class_path", "")),
            None,
        )
        assert jitter_aug is not None
        assert "brightness" in jitter_aug["init_args"]
        assert "contrast" not in jitter_aug["init_args"]

    def test_multiple_augmentations(self, base_config):
        """Test updating multiple augmentations at once."""
        aug_params = {
            "random_affine": {
                "enable": True,
                "max_rotate_degree": 45.0,
            },
            "color_jitter": {
                "enable": True,
                "brightness": 0.3,
            },
            "random_horizontal_flip": {
                "enable": True,
                "probability": 0.6,
            },
        }
        TransformsUpdater.update(aug_params, base_config)

        gpu_augs = base_config["data"]["train_subset"]["augmentations_gpu"]
        class_paths = [a.get("class_path", "") for a in gpu_augs]
        assert any("RandomAffine" in cp for cp in class_paths)
        assert any("ColorJiggle" in cp for cp in class_paths)
        assert any("RandomHorizontalFlip" in cp for cp in class_paths)

    def test_update_tiling_enabled(self, base_config):
        """Test enabling tiling with update_tiling method."""
        tiling_dict = {
            "enable": True,
            "adaptive_tiling": True,
            "tile_size": 800,
            "tile_overlap": 0.2,
        }
        TransformsUpdater.update_tiling(tiling_dict, base_config)

        assert base_config["data"]["tile_config"]["enable_tiler"] is True
        assert base_config["data"]["tile_config"]["enable_adaptive_tiling"] is True
        assert base_config["data"]["tile_config"]["tile_size"] == (800, 800)
        assert base_config["data"]["tile_config"]["overlap"] == 0.2

    def test_update_tiling_disabled(self, base_config):
        """Test disabling tiling with update_tiling method."""
        tiling_dict = {
            "enable": False,
            "adaptive_tiling": False,
            "tile_size": 0,
            "tile_overlap": 0.0,
        }
        TransformsUpdater.update_tiling(tiling_dict, base_config)

        assert base_config["data"]["tile_config"]["enable_tiler"] is False

    def test_update_tiling_none(self, base_config):
        """Test that None tiling dict skips update."""
        original_tiler = base_config["data"]["tile_config"].get("enable_tiler", False)
        TransformsUpdater.update_tiling(None, base_config)

        assert base_config["data"]["tile_config"].get("enable_tiler", False) == original_tiler

    def test_add_random_erasing(self, base_config):
        """Test adding RandomErasing GPU augmentation."""
        aug_params = {
            "random_erasing": {
                "enable": True,
                "scale": [0.02, 0.33],
                "ratio": [0.3, 3.3],
                "probability": 0.5,
            }
        }
        TransformsUpdater.update(aug_params, base_config)

        gpu_augs = base_config["data"]["train_subset"]["augmentations_gpu"]
        erasing_aug = next(
            (a for a in gpu_augs if "RandomErasing" in a.get("class_path", "")),
            None,
        )
        assert erasing_aug is not None
        assert erasing_aug["class_path"] == "kornia.augmentation.RandomErasing"
        assert erasing_aug["init_args"]["scale"] == [0.02, 0.33]
        assert erasing_aug["init_args"]["ratio"] == [0.3, 3.3]
        assert erasing_aug["init_args"]["p"] == 0.5  # probability renamed to p

    def test_add_random_grayscale(self, base_config):
        """Test adding RandomGrayscale GPU augmentation."""
        aug_params = {
            "random_grayscale": {
                "enable": True,
                "probability": 0.2,
            }
        }
        TransformsUpdater.update(aug_params, base_config)

        gpu_augs = base_config["data"]["train_subset"]["augmentations_gpu"]
        gray_aug = next(
            (a for a in gpu_augs if "RandomGrayscale" in a.get("class_path", "")),
            None,
        )
        assert gray_aug is not None
        assert gray_aug["class_path"] == "kornia.augmentation.RandomGrayscale"
        assert gray_aug["init_args"]["p"] == 0.2  # probability renamed to p

    def test_add_random_sharpness(self, base_config):
        """Test adding RandomSharpness GPU augmentation."""
        aug_params = {
            "random_sharpness": {
                "enable": True,
                "sharpness": 0.7,
                "probability": 0.5,
            }
        }
        TransformsUpdater.update(aug_params, base_config)

        gpu_augs = base_config["data"]["train_subset"]["augmentations_gpu"]
        sharp_aug = next(
            (a for a in gpu_augs if "RandomSharpness" in a.get("class_path", "")),
            None,
        )
        assert sharp_aug is not None
        assert sharp_aug["class_path"] == "kornia.augmentation.RandomSharpness"
        assert sharp_aug["init_args"]["sharpness"] == 0.7
        assert sharp_aug["init_args"]["p"] == 0.5  # probability renamed to p

    def test_gaussian_blur(self, base_config):
        """Test that gaussian_blur keeps 'sigma'.
        """
        aug_params = {
            "gaussian_blur": {
                "enable": True,
                "kernel_size": 5,
                "sigma": [0.1, 2.0],
                "probability": 0.5,
            }
        }
        TransformsUpdater.update(aug_params, base_config)

        gpu_augs = base_config["data"]["train_subset"]["augmentations_gpu"]
        blur_aug = next(
            (a for a in gpu_augs if "RandomGaussianBlur" in a.get("class_path", "")),
            None,
        )
        assert blur_aug is not None
        assert "sigma" in blur_aug["init_args"], "sigma must NOT be renamed for gaussian_blur"
        assert "std" not in blur_aug["init_args"], "std must NOT appear for gaussian_blur"
        assert blur_aug["init_args"]["sigma"] == [0.1, 2.0]
        assert blur_aug["init_args"]["p"] == 0.5

    def test_gaussian_noise_sigma(self, base_config):
        """Test that gaussian_noise renames 'sigma' -> 'std' via per-aug param_rename.
        """
        aug_params = {
            "gaussian_noise": {
                "enable": True,
                "mean": 0.0,
                "sigma": 0.15,
                "probability": 0.3,
            }
        }
        TransformsUpdater.update(aug_params, base_config)

        gpu_augs = base_config["data"]["train_subset"]["augmentations_gpu"]
        noise_aug = next(
            (a for a in gpu_augs if "RandomGaussianNoise" in a.get("class_path", "")),
            None,
        )
        assert noise_aug is not None
        assert "std" in noise_aug["init_args"], "sigma must be renamed to std for gaussian_noise"
        assert "sigma" not in noise_aug["init_args"], "sigma must NOT appear in gaussian_noise init_args"
        assert noise_aug["init_args"]["std"] == 0.15
        assert noise_aug["init_args"]["p"] == 0.3

    def test_random_erasing_value_passthrough(self, base_config):
        """Test that RandomErasing 'value' param passes through unchanged."""
        aug_params = {
            "random_erasing": {
                "enable": True,
                "scale": [0.02, 0.33],
                "ratio": [0.3, 3.3],
                "probability": 0.5,
                "value": 0.5,
            }
        }
        TransformsUpdater.update(aug_params, base_config)

        gpu_augs = base_config["data"]["train_subset"]["augmentations_gpu"]
        erasing_aug = next(
            (a for a in gpu_augs if "RandomErasing" in a.get("class_path", "")),
            None,
        )
        assert erasing_aug is not None
        assert erasing_aug["init_args"]["value"] == 0.5
        assert erasing_aug["init_args"]["p"] == 0.5


class TestHyperparametersUpdater:
    """Test HyperparametersUpdater for training hyperparameter updates."""

    @pytest.fixture
    def base_config(self):
        """Create a base config with training-related sections."""
        return {
            "data": {
                "input_size": (640, 640),
                "train_subset": {
                    "batch_size": 32,
                },
                "val_subset": {
                    "batch_size": 32,
                },
            },
            "model": {
                "init_args": {
                    "optimizer": {
                        "init_args": {
                            "lr": 0.001,
                        }
                    }
                }
            },
            "max_epochs": 50,
            "callbacks": [
                {
                    "class_path": "otx.backend.native.callbacks.adaptive_early_stopping.EarlyStoppingWithWarmup",
                    "init_args": {
                        "patience": 10,
                    },
                },
            ],
        }

    def test_update_learning_rate(self, base_config):
        """Test updating learning rate."""
        HyperparametersUpdater.update({"learning_rate": 0.0001}, base_config)

        assert base_config["model"]["init_args"]["optimizer"]["init_args"]["lr"] == 0.0001

    def test_update_batch_size(self, base_config):
        """Test updating batch size for train and val."""
        HyperparametersUpdater.update({"batch_size": 16}, base_config)

        assert base_config["data"]["train_subset"]["batch_size"] == 16
        assert base_config["data"]["val_subset"]["batch_size"] == 16

    def test_update_max_epochs(self, base_config):
        """Test updating max epochs."""
        HyperparametersUpdater.update({"max_epochs": 200}, base_config)

        assert base_config["max_epochs"] == 200

    def test_update_input_size(self, base_config):
        """Test updating input size."""
        HyperparametersUpdater.update({"input_size": (512, 512)}, base_config)

        assert base_config["data"]["input_size"] == (512, 512)

    def test_update_early_stopping_disable(self, base_config):
        """Test disabling early stopping."""
        HyperparametersUpdater.update(
            {"early_stopping": {"enable": False, "patience": 10}},
            base_config,
        )

        # Callback should be removed
        assert not any("EarlyStoppingWithWarmup" in cb.get("class_path", "") for cb in base_config["callbacks"])

    def test_update_early_stopping_enable(self, base_config):
        """Test updating early stopping patience."""
        HyperparametersUpdater.update(
            {"early_stopping": {"enable": True, "patience": 20}},
            base_config,
        )

        callback = next(
            (cb for cb in base_config["callbacks"] if "EarlyStoppingWithWarmup" in cb.get("class_path", "")),
            None,
        )
        assert callback is not None
        assert callback["init_args"]["patience"] == 20

    def test_update_multiple_hyperparams(self, base_config):
        """Test updating multiple hyperparameters at once."""
        HyperparametersUpdater.update(
            {
                "learning_rate": 0.0005,
                "batch_size": 64,
                "max_epochs": 150,
                "input_size": (768, 768),
            },
            base_config,
        )

        assert base_config["model"]["init_args"]["optimizer"]["init_args"]["lr"] == 0.0005
        assert base_config["data"]["train_subset"]["batch_size"] == 64
        assert base_config["data"]["val_subset"]["batch_size"] == 64
        assert base_config["max_epochs"] == 150
        assert base_config["data"]["input_size"] == (768, 768)

    def test_update_with_none_values(self, base_config):
        """Test that None values are skipped."""
        original_config = dict(base_config)

        HyperparametersUpdater.update(
            {
                "learning_rate": None,
                "batch_size": None,
                "input_size": (None, None),
            },
            base_config,
        )

        # Config should remain unchanged
        assert (
            base_config["model"]["init_args"]["optimizer"]["init_args"]["lr"]
            == original_config["model"]["init_args"]["optimizer"]["init_args"]["lr"]
        )
        assert (
            base_config["data"]["train_subset"]["batch_size"] == original_config["data"]["train_subset"]["batch_size"]
        )
        assert base_config["data"]["input_size"] == original_config["data"]["input_size"]
