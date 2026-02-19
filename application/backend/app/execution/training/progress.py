# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any

from lightning import Callback, LightningModule
from lightning import Trainer as LightningTrainer
from lightning.pytorch.utilities.types import STEP_OUTPUT


class TrainingProgressCallback(Callback):
    def __init__(self, report_progress: Callable[[str, float], None], min_p: float = 0, max_p: float = 100.0):
        self._report_progress = report_progress
        self._min_p = min_p
        self._max_p = max_p
        self._total_steps: int | None = None
        self._current_step: int = 0

    def _update_total_steps(self, trainer: LightningTrainer) -> None:
        if self._total_steps is not None:
            return
        max_epochs = trainer.max_epochs or 1
        steps_per_epoch = int(trainer.num_training_batches)
        self._total_steps = max(1, max_epochs * steps_per_epoch)

    def _emit_progress(self) -> None:
        if self._total_steps is None:
            return
        ratio = self._current_step / self._total_steps
        progress = self._min_p + ratio * (self._max_p - self._min_p)
        self._report_progress("", progress)

    def on_train_batch_end(
        self,
        trainer: LightningTrainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._update_total_steps(trainer)
        self._current_step += 1
        self._emit_progress()
