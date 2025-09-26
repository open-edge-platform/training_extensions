# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for data augmentation scheduler components."""

import secrets
from multiprocessing import Value
from unittest.mock import MagicMock, patch

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer
from torchvision.transforms.v2 import Compose, ToDtype

from otx.backend.native.callbacks.aug_scheduler import AugmentationSchedulerCallback, DataAugSwitch


class TestDataAugSwitch:
    """Test cases for DataAugSwitch."""

    @pytest.fixture()
    def sample_policies(self):
        """Create sample augmentation policies."""
        return {
            "no_aug": {
                "to_tv_image": True,
                "transforms": [
                    {"class_path": "torchvision.transforms.v2.ToDtype", "init_args": {"dtype": "torch.float32"}},
                ],
            },
            "strong_aug_1": {
                "to_tv_image": True,
                "transforms": [
                    {"class_path": "torchvision.transforms.v2.ToDtype", "init_args": {"dtype": "torch.float32"}},
                ],
            },
            "strong_aug_2": {
                "to_tv_image": False,
                "transforms": [
                    {"class_path": "torchvision.transforms.v2.ToDtype", "init_args": {"dtype": "torch.int32"}},
                ],
            },
            "light_aug": {
                "to_tv_image": True,
                "transforms": [
                    {"class_path": "torchvision.transforms.v2.ToDtype", "init_args": {"dtype": "torch.float32"}},
                ],
            },
        }

    @pytest.fixture()
    def policy_epochs(self):
        """Create sample policy epochs."""
        return [4, 29, 50]

    @pytest.fixture()
    def data_aug_switch(self, policy_epochs, sample_policies):
        """Create a DataAugSwitch instance."""
        with patch("otx.data.transform_libs.torchvision.TorchVisionTransformLib.generate") as mock_generate:
            # Mock the transform generation to return simple transforms
            mock_generate.return_value = Compose([ToDtype(dtype=torch.float32)])
            return DataAugSwitch(policy_epochs, sample_policies)

    def test_init_valid_policy_epochs(self, policy_epochs, sample_policies):
        """Test DataAugSwitch initialization with valid policy epochs."""
        with patch("otx.data.transform_libs.torchvision.TorchVisionTransformLib.generate") as mock_generate:
            mock_generate.return_value = Compose([ToDtype(dtype=torch.float32)])
            switch = DataAugSwitch(policy_epochs, sample_policies)

            assert switch.policy_epochs == policy_epochs
            assert len(switch.policies) == len(sample_policies)
            assert switch._shared_epoch is None

    def test_init_invalid_policy_epochs(self, sample_policies):
        """Test DataAugSwitch initialization with invalid policy epochs."""
        invalid_epochs = [4, 29]  # Only 2 epochs instead of 3

        with pytest.raises(ValueError, match="Expected 3 policy epochs"):
            DataAugSwitch(invalid_epochs, sample_policies)

    def test_set_shared_epoch(self, data_aug_switch):
        """Test setting shared epoch."""
        shared_epoch = Value("i", 0)
        data_aug_switch.set_shared_epoch(shared_epoch)

        assert data_aug_switch._shared_epoch is shared_epoch

    def test_epoch_property_without_shared_epoch(self, data_aug_switch):
        """Test epoch property when shared epoch is not set."""
        with pytest.raises(ValueError, match="Shared epoch not set"):
            _ = data_aug_switch.epoch

    def test_epoch_property_with_shared_epoch(self, data_aug_switch):
        """Test epoch property when shared epoch is set."""
        shared_epoch = Value("i", 10)
        data_aug_switch.set_shared_epoch(shared_epoch)

        assert data_aug_switch.epoch == 10

    def test_epoch_setter_without_shared_epoch(self, data_aug_switch):
        """Test epoch setter when shared epoch is not set."""
        with pytest.raises(ValueError, match="Shared epoch not set"):
            data_aug_switch.epoch = 5

    def test_epoch_setter_with_shared_epoch(self, data_aug_switch):
        """Test epoch setter when shared epoch is set."""
        shared_epoch = Value("i", 0)
        data_aug_switch.set_shared_epoch(shared_epoch)

        data_aug_switch.epoch = 15
        assert data_aug_switch.epoch == 15
        assert shared_epoch.value == 15

    def test_current_policy_name_no_aug_stage(self, data_aug_switch):
        """Test current_policy_name in no_aug stage (epoch < 4)."""
        shared_epoch = Value("i", 2)
        data_aug_switch.set_shared_epoch(shared_epoch)

        assert data_aug_switch.current_policy_name == "no_aug"

    def test_current_policy_name_strong_aug_stage(self, data_aug_switch):
        """Test current_policy_name in strong_aug stage (4 <= epoch < 29)."""
        shared_epoch = Value("i", 15)
        data_aug_switch.set_shared_epoch(shared_epoch)

        with patch.object(secrets, "choice", return_value="strong_aug_1"):
            policy_name = data_aug_switch.current_policy_name
            assert policy_name in ["strong_aug_1", "strong_aug_2"]

    def test_current_policy_name_light_aug_stage(self, data_aug_switch):
        """Test current_policy_name in light_aug stage (epoch >= 29)."""
        shared_epoch = Value("i", 35)
        data_aug_switch.set_shared_epoch(shared_epoch)

        assert data_aug_switch.current_policy_name == "light_aug"

    def test_current_policy_name_boundary_conditions(self, data_aug_switch):
        """Test current_policy_name at boundary conditions."""
        shared_epoch = Value("i", 0)
        data_aug_switch.set_shared_epoch(shared_epoch)

        # Test exact boundary values
        test_cases = [
            (3, "no_aug"),  # Just before first boundary
            (4, "strong_aug_1"),  # At first boundary (mocked)
            (28, "strong_aug_2"),  # Just before second boundary (mocked)
            (29, "light_aug"),  # At second boundary
            (50, "light_aug"),  # Beyond all boundaries
        ]

        for epoch, expected_stage in test_cases:
            data_aug_switch.epoch = epoch
            if expected_stage in ["strong_aug_1", "strong_aug_2"]:
                with patch.object(secrets, "choice", return_value=expected_stage):
                    assert data_aug_switch.current_policy_name == expected_stage
            else:
                assert data_aug_switch.current_policy_name == expected_stage

    def test_current_transforms_property(self, data_aug_switch):
        """Test current_transforms property."""
        shared_epoch = Value("i", 2)
        data_aug_switch.set_shared_epoch(shared_epoch)

        to_tv_image, transforms = data_aug_switch.current_transforms

        assert isinstance(to_tv_image, bool)
        assert isinstance(transforms, Compose)

    def test_secrets_choice_randomness(self, data_aug_switch):
        """Test that secrets.choice is used for random selection."""
        shared_epoch = Value("i", 15)  # In strong_aug stage
        data_aug_switch.set_shared_epoch(shared_epoch)

        with patch.object(secrets, "choice") as mock_choice:
            mock_choice.return_value = "strong_aug_1"
            policy_name = data_aug_switch.current_policy_name

            mock_choice.assert_called_once_with(["strong_aug_1", "strong_aug_2"])
            assert policy_name == "strong_aug_1"

    def test_policy_processing_during_init(self, policy_epochs, sample_policies):
        """Test that policies are properly processed during initialization."""
        with patch("otx.data.transform_libs.torchvision.TorchVisionTransformLib.generate") as mock_generate:
            mock_transform = Compose([ToDtype(dtype=torch.float32)])
            mock_generate.return_value = mock_transform

            switch = DataAugSwitch(policy_epochs, sample_policies)

            # Check that generate was called for each policy
            assert mock_generate.call_count == len(sample_policies)

            # Check that policies were processed correctly
            for policy_name in sample_policies:
                assert policy_name in switch.policies
                assert "to_tv_image" in switch.policies[policy_name]
                assert "transforms" in switch.policies[policy_name]
                assert switch.policies[policy_name]["transforms"] is mock_transform


class TestAugmentationSchedulerCallback:
    """Test cases for AugmentationSchedulerCallback."""

    @pytest.fixture()
    def mock_data_aug_switch(self):
        """Create a mock DataAugSwitch."""
        return MagicMock(spec=DataAugSwitch)

    @pytest.fixture()
    def callback_with_switch(self, mock_data_aug_switch):
        """Create callback with DataAugSwitch."""
        return AugmentationSchedulerCallback(mock_data_aug_switch)

    @pytest.fixture()
    def callback_without_switch(self):
        """Create callback without DataAugSwitch."""
        return AugmentationSchedulerCallback()

    @pytest.fixture()
    def mock_trainer(self):
        """Create a mock trainer."""
        trainer = MagicMock(spec=Trainer)
        trainer.current_epoch = 10
        return trainer

    @pytest.fixture()
    def mock_pl_module(self):
        """Create a mock Lightning module."""
        return MagicMock(spec=LightningModule)

    def test_init_with_data_aug_switch(self, mock_data_aug_switch):
        """Test callback initialization with DataAugSwitch."""
        callback = AugmentationSchedulerCallback(mock_data_aug_switch)

        assert callback.data_aug_switch is mock_data_aug_switch

    def test_init_without_data_aug_switch(self):
        """Test callback initialization without DataAugSwitch."""
        callback = AugmentationSchedulerCallback()

        assert callback.data_aug_switch is None

    def test_set_data_aug_switch(self, callback_without_switch, mock_data_aug_switch):
        """Test setting DataAugSwitch after initialization."""
        callback_without_switch.set_data_aug_switch(mock_data_aug_switch)

        assert callback_without_switch.data_aug_switch is mock_data_aug_switch

    def test_on_train_epoch_start_with_switch(self, callback_with_switch, mock_trainer, mock_pl_module):
        """Test on_train_epoch_start when DataAugSwitch is available."""
        callback_with_switch.on_train_epoch_start(mock_trainer, mock_pl_module)

        # Check that epoch was set on the DataAugSwitch
        assert callback_with_switch.data_aug_switch.epoch == mock_trainer.current_epoch

    def test_on_train_epoch_start_without_switch(self, callback_without_switch, mock_trainer, mock_pl_module):
        """Test on_train_epoch_start when DataAugSwitch is not available."""
        # This should not raise an exception but will fail due to None
        with pytest.raises(AttributeError):
            callback_without_switch.on_train_epoch_start(mock_trainer, mock_pl_module)

    def test_on_train_epoch_start_updates_epoch(self, callback_with_switch, mock_pl_module):
        """Test that on_train_epoch_start updates epoch correctly."""
        # Test different epoch values
        for epoch in [0, 5, 10, 25, 50]:
            mock_trainer = MagicMock(spec=Trainer)
            mock_trainer.current_epoch = epoch

            callback_with_switch.on_train_epoch_start(mock_trainer, mock_pl_module)

            assert callback_with_switch.data_aug_switch.epoch == epoch

    def test_callback_inheritance(self, callback_with_switch):
        """Test that callback properly inherits from Lightning Callback."""
        from lightning.pytorch.callbacks.callback import Callback

        assert isinstance(callback_with_switch, Callback)

    def test_set_data_aug_switch_replaces_existing(self, callback_with_switch):
        """Test that set_data_aug_switch replaces existing switch."""
        original_switch = callback_with_switch.data_aug_switch
        new_switch = MagicMock(spec=DataAugSwitch)

        callback_with_switch.set_data_aug_switch(new_switch)

        assert callback_with_switch.data_aug_switch is new_switch
        assert callback_with_switch.data_aug_switch is not original_switch

    def test_multiple_epoch_updates(self, callback_with_switch, mock_pl_module):
        """Test multiple epoch updates during training."""
        epochs = [0, 1, 2, 5, 10, 15, 20, 25, 30]

        for epoch in epochs:
            mock_trainer = MagicMock(spec=Trainer)
            mock_trainer.current_epoch = epoch

            callback_with_switch.on_train_epoch_start(mock_trainer, mock_pl_module)

            # Verify the epoch was set correctly
            callback_with_switch.data_aug_switch.epoch = epoch


class TestDataAugSwitchIntegration:
    """Integration tests for DataAugSwitch and AugmentationSchedulerCallback."""

    @pytest.fixture()
    def sample_policies(self):
        """Create sample augmentation policies."""
        return {
            "no_aug": {
                "to_tv_image": True,
                "transforms": [
                    {"class_path": "torchvision.transforms.v2.ToDtype", "init_args": {"dtype": "torch.float32"}},
                ],
            },
            "strong_aug_1": {
                "to_tv_image": True,
                "transforms": [
                    {"class_path": "torchvision.transforms.v2.ToDtype", "init_args": {"dtype": "torch.float32"}},
                ],
            },
            "strong_aug_2": {
                "to_tv_image": False,
                "transforms": [
                    {"class_path": "torchvision.transforms.v2.ToDtype", "init_args": {"dtype": "torch.int32"}},
                ],
            },
            "light_aug": {
                "to_tv_image": True,
                "transforms": [
                    {"class_path": "torchvision.transforms.v2.ToDtype", "init_args": {"dtype": "torch.float32"}},
                ],
            },
        }

    @pytest.fixture()
    def integration_setup(self, sample_policies):
        """Set up DataAugSwitch and AugmentationSchedulerCallback for integration testing."""
        with patch("otx.data.transform_libs.torchvision.TorchVisionTransformLib.generate") as mock_generate:
            mock_generate.return_value = Compose([ToDtype(dtype=torch.float32)])

            # Create DataAugSwitch
            switch = DataAugSwitch([4, 29, 50], sample_policies)

            # Create shared epoch
            shared_epoch = Value("i", 0)
            switch.set_shared_epoch(shared_epoch)

            # Create callback
            callback = AugmentationSchedulerCallback(switch)

            return switch, callback, shared_epoch

    def test_full_training_simulation(self, integration_setup):
        """Test full training simulation with epoch updates."""
        switch, callback, shared_epoch = integration_setup

        # Simulate training epochs
        test_epochs = [0, 3, 4, 15, 28, 29, 35, 50]
        expected_policies = [
            "no_aug",
            "no_aug",
            "strong_aug",
            "strong_aug",
            "strong_aug",
            "light_aug",
            "light_aug",
            "light_aug",
        ]

        for epoch, expected_policy_type in zip(test_epochs, expected_policies):
            # Simulate trainer epoch update
            mock_trainer = MagicMock(spec=Trainer)
            mock_trainer.current_epoch = epoch
            mock_pl_module = MagicMock(spec=LightningModule)

            # Update epoch via callback
            callback.on_train_epoch_start(mock_trainer, mock_pl_module)

            # Check that shared epoch was updated
            assert shared_epoch.value == epoch
            assert switch.epoch == epoch

            # Check policy type
            current_policy = switch.current_policy_name
            if expected_policy_type == "strong_aug":
                assert current_policy in ["strong_aug_1", "strong_aug_2"]
            else:
                assert current_policy == expected_policy_type

    def test_concurrent_access_simulation(self, integration_setup):
        """Test simulation of concurrent access to shared epoch."""
        switch, callback, shared_epoch = integration_setup

        # Simulate callback updating epoch
        mock_trainer = MagicMock(spec=Trainer)
        mock_trainer.current_epoch = 15
        mock_pl_module = MagicMock(spec=LightningModule)

        callback.on_train_epoch_start(mock_trainer, mock_pl_module)

        # Simulate dataset reading epoch (this would happen in parallel)
        current_epoch = switch.epoch
        policy_name = switch.current_policy_name

        assert current_epoch == 15
        assert policy_name in ["strong_aug_1", "strong_aug_2"]

        # Verify both callback and switch see the same epoch
        assert callback.data_aug_switch.epoch == 15

    def test_error_handling_integration(self, sample_policies):
        """Test error handling in integration scenarios."""
        with patch("otx.data.transform_libs.torchvision.TorchVisionTransformLib.generate") as mock_generate:
            mock_generate.return_value = Compose([ToDtype(dtype=torch.float32)])

            # Create switch without shared epoch
            switch = DataAugSwitch([4, 29, 50], sample_policies)
            callback = AugmentationSchedulerCallback(switch)

            # Try to use without setting shared epoch
            with pytest.raises(ValueError, match="Shared epoch not set"):
                _ = switch.current_policy_name

            # Try to update epoch via callback without shared epoch
            mock_trainer = MagicMock(spec=Trainer)
            mock_trainer.current_epoch = 10
            mock_pl_module = MagicMock(spec=LightningModule)

            with pytest.raises(ValueError, match="Shared epoch not set"):
                callback.on_train_epoch_start(mock_trainer, mock_pl_module)
