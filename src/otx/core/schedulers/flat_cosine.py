# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""FlatCosineScheduler schedulers."""
from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING

from torch.optim.lr_scheduler import LRScheduler

if TYPE_CHECKING:
    from torch.optim.optimizer import Optimizer


def flat_cosine_schedule(
    decay_iters: int,
    warmup_iters: int,
    flat_lr_iters: int,
    no_aug_iters: int,
    current_iter: int,
    init_lr: float,
    min_lr: float,
) -> float:
    """Computes the learning rate using a warm-up, flat, and cosine decay schedule.

    Args:
        decay_iters (int): Total number of iterations.
        warmup_iter (int): Number of iterations for warm-up phase.
        flat_lr_iters (int): Number of iterations for flat phase.
        no_aug_iters (int): Number of iterations for no-augmentation phase.
        current_iter (int): Current iteration.
        init_lr (float): Initial learning rate.
        min_lr (float): Minimum learning rate.

    Returns:
        float: Calculated learning rate.
    """
    if current_iter <= warmup_iters:
        return init_lr * (current_iter / float(warmup_iters)) ** 2
    if warmup_iters < current_iter <= flat_lr_iters:
        return init_lr
    if current_iter >= decay_iters - no_aug_iters:
        return min_lr

    cosine_decay = 0.5 * (
        1 + math.cos(math.pi * (current_iter - flat_lr_iters) / (decay_iters - flat_lr_iters - no_aug_iters))
    )
    return min_lr + (init_lr - min_lr) * cosine_decay


class FlatCosineScheduler(LRScheduler):
    """Flat Cosine Scheduler.

    This scheduler implements a flat cosine learning rate schedule, where the learning rate remains constant
    for a specified number of steps before decaying according to a cosine function.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        iter_per_epoch (int): Number of iterations per epoch.
        lr_gamma (float): Factor by which the learning rate is multiplied at the end of the schedule.
        decay_duration (int): Total duration of the annealing phase in epochs.
        warmup_iters (int): Number of iterations for the warm-up phase.
        flat_lr_duration (int): Duration of the flat learning rate phase in epochs.
        no_aug_epochs (int): Number of epochs without augmentation.
        interval (str): Interval for updating the learning rate, either "step" or "epoch".
    """

    def __init__(
        self,
        optimizer: Optimizer,
        iter_per_epoch: int,
        lr_gamma: float = 0.5,
        decay_duration: int = 40,
        warmup_iters: int = 30,
        flat_lr_duration: int = 23,
        no_aug_epochs: int = 8,
        interval: str = "step",
    ) -> None:
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.min_lrs = [base_lr * lr_gamma for base_lr in self.base_lrs]
        self.interval = interval

        decay_iters = int(iter_per_epoch * decay_duration)
        no_aug_iters = int(iter_per_epoch * no_aug_epochs)
        flat_lr_iters = int(iter_per_epoch * flat_lr_duration)

        self.lr_func = partial(
            flat_cosine_schedule,
            decay_iters,
            warmup_iters,
            flat_lr_iters,
            no_aug_iters,
        )

        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        """Compute the learning rate for the current epoch."""
        return [
            self.lr_func(self._step_count, self.base_lrs[i], self.min_lrs[i])
            for i in range(len(self.optimizer.param_groups))
        ]
