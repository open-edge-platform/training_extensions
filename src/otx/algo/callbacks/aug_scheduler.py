# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Data augmentation scheduler for training."""
from __future__ import annotations

import random
from multiprocessing import Value
from typing import Any

from lightning.pytorch.callbacks.callback import Callback

from otx.core.config.data import SubsetConfig
from otx.core.data.transform_libs.torchvision import Compose, TorchVisionTransformLib


class DataAugSwitch:
    def __init__(
        self,
        policy_epochs,
        policies: dict[str, Any],
    ) -> None:
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

    def set_shared_epoch(self, shared_epoch: Value) -> None:
        self._shared_epoch = shared_epoch

    @property
    def epoch(self):
        if not hasattr(self, "_shared_epoch"):
            raise ValueError("Shared epoch not set. Call set_shared_epoch() first.")
        return self._shared_epoch.value

    @epoch.setter
    def epoch(self, value):
        if not hasattr(self, "_shared_epoch"):
            raise ValueError("Shared epoch not set. Call set_shared_epoch() first.")
        self._shared_epoch.value = value

    @property
    def current_policy_name(self) -> str:
        e = self.epoch
        p0, p1, _ = self.policy_epochs
        if e < p0:
            return "no_aug"
        if p0 <= e < p1:
            return "strong_aug_1" if random.random() < 0.5 else "strong_aug_2"

        return "light_aug"

    @property
    def current_transforms(self) -> tuple[bool, Compose]:
        name = self.current_policy_name
        policy = self.policies.get(name)
        return policy["to_tv_image"], policy["transforms"]


class AugmentationSchedulerCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.data_aug_switch.epoch = trainer.current_epoch

    def set_data_aug_switch(self, data_aug_switch: DataAugSwitch) -> None:
        """Set data augmentation switch."""
        self.data_aug_switch = data_aug_switch
