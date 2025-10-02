# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for dataset mixins."""

from unittest.mock import MagicMock

import pytest
import torch
from torchvision.transforms.v2 import Compose, ToDtype

from otx.data.dataset.mixins import DataAugSwitchMixin
from otx.data.entity.torch import OTXDataItem


class MockDataset(DataAugSwitchMixin):
    """Mock dataset class for testing the mixin."""

    def __init__(self, *args, **kwargs):
        self.to_tv_image = True
        self.transforms = None

    def _apply_transforms(self, entity: OTXDataItem) -> OTXDataItem:
        return entity


class TestDataAugSwitchMixin:
    """Test cases for DataAugSwitchMixin."""

    @pytest.fixture()
    def mock_dataset(self):
        """Create a mock dataset with the mixin."""
        return MockDataset()

    @pytest.fixture()
    def mock_data_aug_switch(self):
        """Create a mock DataAugSwitch."""
        mock_switch = MagicMock()
        mock_transforms = Compose([ToDtype(dtype=torch.float32)])
        mock_switch.current_transforms = (True, mock_transforms)
        return mock_switch

    @pytest.fixture()
    def mock_entity(self):
        """Create a mock OTXDataItem."""
        return MagicMock(spec=OTXDataItem)

    def test_lazy_initialization(self, mock_dataset):
        """Test that mixin initializes lazily."""
        # Initially, the attribute shouldn't exist
        assert not hasattr(mock_dataset, "data_aug_switch")

        # After calling has_dynamic_augmentation, it should be initialized
        assert not mock_dataset.has_dynamic_augmentation
        assert hasattr(mock_dataset, "data_aug_switch")
        assert mock_dataset.data_aug_switch is None

    def test_set_data_aug_switch(self, mock_dataset, mock_data_aug_switch):
        """Test setting data augmentation switch."""
        mock_dataset.set_data_aug_switch(mock_data_aug_switch)
        assert mock_dataset.data_aug_switch is mock_data_aug_switch

    def test_has_dynamic_augmentation_false_when_none(self, mock_dataset):
        """Test has_dynamic_augmentation returns False when no switch is set."""
        assert not mock_dataset.has_dynamic_augmentation

    def test_has_dynamic_augmentation_true_when_set(self, mock_dataset, mock_data_aug_switch):
        """Test has_dynamic_augmentation returns True when switch is set."""
        mock_dataset.set_data_aug_switch(mock_data_aug_switch)
        assert mock_dataset.has_dynamic_augmentation

    def test_apply_augmentation_switch_with_switch(self, mock_dataset, mock_data_aug_switch, mock_entity):
        """Test _apply_augmentation_switch when switch is set."""
        mock_dataset.set_data_aug_switch(mock_data_aug_switch)

        policy_name = mock_dataset._apply_augmentation_switch()

        assert mock_dataset.to_tv_image is mock_data_aug_switch.policies[policy_name]["to_tv_image"]
        assert mock_dataset.transforms is mock_data_aug_switch.policies[policy_name]["transforms"]

    def test_apply_augmentation_switch_updates_transforms(self, mock_dataset, mock_entity):
        """Test that augmentation switch properly updates transforms."""
        # Create a mock switch with specific transforms
        mock_switch = MagicMock()
        new_transforms = Compose([ToDtype(dtype=torch.int32)])
        mock_switch.current_transforms = (False, new_transforms)

        mock_dataset.set_data_aug_switch(mock_switch)
        policy_name = mock_dataset._apply_augmentation_switch()

        assert mock_dataset.to_tv_image is mock_switch.policies[policy_name]["to_tv_image"]
        assert mock_dataset.transforms is mock_switch.policies[policy_name]["transforms"]

    def test_multiple_switch_updates(self, mock_dataset):
        """Test multiple updates to the augmentation switch."""
        # First switch
        mock_switch1 = MagicMock()
        transforms1 = Compose([ToDtype(dtype=torch.float32)])
        mock_switch1.current_transforms = (True, transforms1)

        mock_dataset.set_data_aug_switch(mock_switch1)
        policy_name = mock_dataset._apply_augmentation_switch()

        assert mock_dataset.to_tv_image is mock_switch1.policies[policy_name]["to_tv_image"]
        assert mock_dataset.transforms is mock_switch1.policies[policy_name]["transforms"]

        # Second switch
        mock_switch2 = MagicMock()
        transforms2 = Compose([ToDtype(dtype=torch.int32)])
        mock_switch2.current_transforms = (False, transforms2)

        mock_dataset.set_data_aug_switch(mock_switch2)
        policy_name = mock_dataset._apply_augmentation_switch()

        assert mock_dataset.to_tv_image is mock_switch2.policies[policy_name]["to_tv_image"]
        assert mock_dataset.transforms is mock_switch2.policies[policy_name]["transforms"]

    def test_has_dynamic_augmentation_property_edge_cases(self):
        """Test edge cases for has_dynamic_augmentation property."""

        # Dataset without the attribute (should be lazily initialized)
        class DatasetWithoutSwitch:
            pass

        dataset = DatasetWithoutSwitch()
        dataset._ensure_data_aug_switch_initialized = DataAugSwitchMixin._ensure_data_aug_switch_initialized.__get__(
            dataset,
        )
        dataset.has_dynamic_augmentation = DataAugSwitchMixin.has_dynamic_augmentation.__get__(dataset)

        assert not dataset.has_dynamic_augmentation
        # After calling has_dynamic_augmentation, the attribute should be initialized
        assert hasattr(dataset, "data_aug_switch")
        assert dataset.data_aug_switch is None

        # Dataset with None value
        class DatasetWithNoneSwitch:
            def __init__(self):
                self.data_aug_switch = None

        dataset2 = DatasetWithNoneSwitch()
        dataset2._ensure_data_aug_switch_initialized = DataAugSwitchMixin._ensure_data_aug_switch_initialized.__get__(
            dataset2,
        )
        dataset2.has_dynamic_augmentation = DataAugSwitchMixin.has_dynamic_augmentation.__get__(dataset2)

        assert not dataset2.has_dynamic_augmentation
