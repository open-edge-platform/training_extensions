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
    """Data augmentation switch for dynamic scheduling of augmentation policies during training.

    This class manages multiple data augmentation policies and switches between them
    based on the current training epoch. It is designed to support multi-stage augmentation
    strategies, such as starting with no augmentation, then applying strong augmentations,
    and finally switching to lighter augmentations as training progresses.

    The switch is typically used in conjunction with a callback (e.g., AugmentationSchedulerCallback)
    that updates the current epoch, allowing the augmentation policy to change automatically
    as training advances.

    Args:
        policy_epochs (list[int]): List of 3 epoch indices that define the boundaries between
            augmentation stages. For example, [4, 29, 50] means:
                - Epochs < 4: use "no_aug"
                - 4 <= epochs < 29: use "strong_aug_1" or "strong_aug_2" (randomly chosen)
                - epochs >= 29: use "light_aug"
        policies (dict[str, dict[str, Any]]): Dictionary mapping policy names to their configuration.
            Each configuration should include a "transforms" key (list of transform configs),
            and optionally "to_tv_image" (bool).

    Attributes:
        policy_epochs (list[int]): The epoch boundaries for switching policies.
        policies (dict[str, dict[str, Any]]): The processed policy configurations.
        _shared_epoch: A multiprocessing.Value or similar object for sharing the current epoch.
            This attribute holds a reference to a multiprocessing.Value used to synchronize
            and share the current training epoch across multiple processes.
            This is necessary in distributed or multi-process training scenarios, where each process may
            need to access or update the current epoch in a thread-safe and consistent manner.
            By using a shared object, the augmentation policy can be switched reliably based on the global
            training progress, ensuring all processes use the correct augmentation strategy.

    Example:
        >>> policy_epochs = [4, 29, 50]
        >>> policies = {
        ...     "no_aug": {"transforms": [...]},
        ...     "strong_aug_1": {"transforms": [...]},
        ...     "strong_aug_2": {"transforms": [...]},
        ...     "light_aug": {"transforms": [...]},
        ... }
        >>> switch = DataAugSwitch(policy_epochs, policies)
        >>> switch.set_shared_epoch(shared_epoch)
        >>> # During training, update epoch:
        >>> switch.epoch = 10
        >>> to_tv_image, transforms = switch.current_transforms

    Note:
        - The current policy is determined by the current epoch and the provided policy_epochs.
        - For the "strong augmentation" stage, one of the strong policies is randomly selected
          for each call using a cryptographically secure random choice.
        - The transforms for each policy are generated using TorchVisionTransformLib.

    """

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
    """Callback for managing data augmentation scheduling during training.

    This callback is designed to work with a `DataAugSwitch` object, which controls
    the augmentation policy applied to the training data at each epoch. The callback
    updates the current epoch in the `DataAugSwitch` at the start of each training epoch,
    allowing the augmentation policy to change dynamically as training progresses.

    Typical usage involves attaching this callback to a PyTorch Lightning Trainer,
    and providing it with a `DataAugSwitch` instance that manages the augmentation logic.

    Args:
        data_aug_switch (DataAugSwitch | None): Optional. The DataAugSwitch instance
            that controls augmentation policies. Can be set later via `set_data_aug_switch()`.

    Example:
        >>> data_aug_switch = DataAugSwitch(...)
        >>> aug_callback = AugmentationSchedulerCallback(data_aug_switch)
        >>> trainer = Trainer(callbacks=[aug_callback])
        >>> trainer.fit(model, datamodule=...)

        # Alternatively, set the DataAugSwitch after instantiation:
        >>> aug_callback = AugmentationSchedulerCallback()
        >>> aug_callback.set_data_aug_switch(data_aug_switch)
        >>> trainer = Trainer(callbacks=[aug_callback])

    Note:
        - The `DataAugSwitch` instance must be set before training starts.
        - This callback assumes that the `DataAugSwitch` object has an `epoch` property
          that can be updated to reflect the current training epoch.
    """

    def __init__(self, data_aug_switch: DataAugSwitch | None = None):
        super().__init__()
        self.data_aug_switch = data_aug_switch

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Update the DataAugSwitch with the current epoch at the start of each training epoch.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer instance.
            pl_module (LightningModule): The LightningModule being trained.
        """
        self.data_aug_switch.epoch = trainer.current_epoch  # type: ignore[union-attr]

    def set_data_aug_switch(self, data_aug_switch: DataAugSwitch) -> None:
        """Set or update the DataAugSwitch instance for this callback.

        Args:
            data_aug_switch (DataAugSwitch): The DataAugSwitch to use.
        """
        self.data_aug_switch = data_aug_switch
