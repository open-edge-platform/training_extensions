# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Data augmentation scheduler for training."""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any

from lightning.pytorch.callbacks.callback import Callback

from otx.config.data import SubsetConfig
from otx.data.transform_libs.torchvision import Compose, TorchVisionTransformLib

if TYPE_CHECKING:
    from multiprocessing import Value

    from lightning.pytorch import LightningModule, Trainer


class DataAugSwitch:
    """Data augmentation switch."""

    def __init__(
        self,
        policy_epochs: list[int],
        policies: dict[str, dict[str, Any]],
    ) -> None:
        """Initialize the data augmentation switch."""
        if len(policy_epochs) != 3:
            msg = "Expected 3 policy epochs for 4-stage scheduler (e.g., [4, 29, 50])"
            raise ValueError(msg)

        self.policy_epochs = policy_epochs
        self.policies = policies
        self._shared_epoch = None

        # Compose transforms for each policy
        for name, config in policies.items():
            self.policies[name] = {
                "to_tv_image": config.get("to_tv_image", True),
                "transforms": TorchVisionTransformLib.generate(
                    config=SubsetConfig(
                        transforms=config["transforms"],
                        batch_size=1,
                        subset_name=name,
                    ),
                ),
            }

    def set_shared_epoch(self, shared_epoch: Value) -> None:  # type: ignore[valid-type]
        """Set the shared epoch."""
        self._shared_epoch = shared_epoch

    @property
    def epoch(self) -> int:
        """Get the current epoch."""
        if self._shared_epoch is None:
            msg = "Shared epoch not set. Call set_shared_epoch() first."
            raise ValueError(msg)
        return self._shared_epoch.value

    @epoch.setter
    def epoch(self, value: int) -> None:
        """Set the current epoch."""
        if self._shared_epoch is None:
            msg = "Shared epoch not set. Call set_shared_epoch() first."
            raise ValueError(msg)
        self._shared_epoch.value = value

    @property
    def current_policy_name(self) -> str:
        """Get the current policy name."""
        e = self.epoch
        p0, p1, _ = self.policy_epochs
        if e < p0:
            return "no_aug"
        if p0 <= e < p1:
            # Use secrets.choice for cryptographically secure random selection
            return secrets.choice(["strong_aug_1", "strong_aug_2"])
        return "light_aug"

    @property
    def current_transforms(self) -> tuple[bool, Compose]:
        """Get the current transforms."""
        name = self.current_policy_name
        policy = self.policies.get(name)
        return policy["to_tv_image"], policy["transforms"]  # type: ignore[index]


class AugmentationSchedulerCallback(Callback):
    """Callback for data augmentation scheduler."""

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Set the current epoch."""
        self.data_aug_switch.epoch = trainer.current_epoch

    def set_data_aug_switch(self, data_aug_switch: DataAugSwitch) -> None:
        """Set data augmentation switch."""
        self.data_aug_switch = data_aug_switch
