# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mixins for OTX datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from otx.backend.native.callbacks.aug_scheduler import DataAugSwitch


class DataAugSwitchMixin:
    """Mixin class that provides dynamic augmentation switching functionality.

    This mixin can be used by any dataset that needs to switch between different
    augmentation policies during training based on epoch information.

    Usage:
        class MyDataset(OTXDataset, DataAugSwitchMixin):
            def _get_item_impl(self, index: int) -> OTXDataItem | None:
                # ... get your data ...
                self._apply_augmentation_switch()
                return self._apply_transforms(entity)
    """

    def _ensure_data_aug_switch_initialized(self) -> None:
        """Ensure data_aug_switch attribute is initialized.

        This method is called lazily since __init__ may not be called
        due to multiple inheritance MRO in some dataset classes.
        """
        if not hasattr(self, "data_aug_switch"):
            self.data_aug_switch: DataAugSwitch | None = None

    def set_data_aug_switch(self, data_aug_switch: DataAugSwitch) -> None:
        """Set data augmentation switch.

        Args:
            data_aug_switch: DataAugSwitch instance that manages dynamic augmentation policies
        """
        self._ensure_data_aug_switch_initialized()
        self.data_aug_switch = data_aug_switch

    def _apply_augmentation_switch(self) -> str | None:
        """Update the dataset's transform configuration based on the current augmentation policy.

        This method should be called before applying the regular transforms.
        It updates the dataset's transform configuration based on the current
        augmentation policy from DataAugSwitch, if available.

        Returns:
            str | None: The name of the current policy, or None if no policy is set.
        """
        self._ensure_data_aug_switch_initialized()
        if self.data_aug_switch is None:
            return None
        policy_name = self.data_aug_switch.current_policy_name
        policy = self.data_aug_switch.policies[policy_name]
        self.to_tv_image, self.transforms = policy["to_tv_image"], policy["transforms"]
        return policy_name

    @property
    def has_dynamic_augmentation(self) -> bool:
        """Check if dynamic augmentation is available and configured."""
        self._ensure_data_aug_switch_initialized()
        return self.data_aug_switch is not None
