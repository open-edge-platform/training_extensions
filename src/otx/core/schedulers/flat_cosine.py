# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""FlatCosineScheduler schedulers"""
from torch.optim.lr_scheduler import LRScheduler


import math
from functools import partial


def flat_cosine_schedule(
    total_iters, 
    warmup_iters, 
    flat_iters, 
    no_aug_iters, 
    current_iter, 
    init_lr, 
    min_lr
):
    """
    Computes the learning rate using a warm-up, flat, and cosine decay schedule.

    Args:
        total_iter (int): Total number of iterations.
        warmup_iter (int): Number of iterations for warm-up phase.
        flat_iter (int): Number of iterations for flat phase.
        no_aug_iter (int): Number of iterations for no-augmentation phase.
        current_iter (int): Current iteration.
        init_lr (float): Initial learning rate.
        min_lr (float): Minimum learning rate.

    Returns:
        float: Calculated learning rate.
    """
    if current_iter <= warmup_iters:
        return init_lr * (current_iter / float(warmup_iters)) ** 2
    elif warmup_iters < current_iter <= flat_iters:
        return init_lr
    elif current_iter >= total_iters - no_aug_iters:
        return min_lr
    else:
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (current_iter - flat_iters) /
                                           (total_iters - flat_iters - no_aug_iters)))
        return min_lr + (init_lr - min_lr) * cosine_decay

class FlatCosineScheduler(LRScheduler):
    """Flat Cosine Scheduler.

    This scheduler implements a flat cosine learning rate schedule, where the learning rate remains constant
    for a specified number of steps before decaying according to a cosine function.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        num_flat_steps (int): Number of steps during which the learning rate remains constant.
        num_decay_steps (int): Number of steps over which the learning rate decays.
        last_epoch (int, optional): The index of the last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer,
        iter_per_epoch,
        lr_gamma: float = 0.5,
        total_epochs: float = 40, 
        warmup_iters: float = 30,
        flat_epochs: float = -1, 
        no_aug_epochs: float = -1,
        interval: str = "step",
    ):
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.min_lrs = [base_lr * lr_gamma for base_lr in self.base_lrs]
        self.interval = interval

        self.lr_gamma = lr_gamma
        self.no_aug_epochs = no_aug_epochs
        self.flat_epochs = flat_epochs
        self.total_epochs = total_epochs
        self.warmup_iters = warmup_iters

        total_iters = int(iter_per_epoch * self.total_epochs)
        no_aug_iters = int(iter_per_epoch * self.no_aug_epochs)
        flat_iters = int(iter_per_epoch * self.flat_epochs)

        self.lr_func = partial(
            flat_cosine_schedule, 
            total_iters,
            self.warmup_iters,
            flat_iters,
            no_aug_iters
        )

        super().__init__(optimizer)

    def get_lr(self):
        """Compute the learning rate for the current epoch."""
        return [
            self.lr_func(self._step_count, self.base_lrs[i], self.min_lrs[i]) 
            for i, group in enumerate(self.optimizer.param_groups)
        ]
        