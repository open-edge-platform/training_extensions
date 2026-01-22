# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

from lightning import Trainer as LightningTrainer

from app.services.training.otx_trainer import TrainingProgressCallback


class DummyTrainer:
    def __init__(self, max_epochs: int | None, num_training_batches: int):
        self.max_epochs = max_epochs
        self.num_training_batches = num_training_batches


def _make_trainer(max_epochs: int | None = 5, num_training_batches: int = 10) -> LightningTrainer:
    return DummyTrainer(max_epochs=max_epochs, num_training_batches=num_training_batches)  # type: ignore[return-value]


class TestTrainingProgressCallback:
    def test_emit_progress_does_nothing_when_total_steps_none(self) -> None:
        report_progress = Mock()
        cb = TrainingProgressCallback(report_progress)

        cb._emit_progress()

        report_progress.assert_not_called()

    def test_on_train_batch_end_emits_progress_in_range(self) -> None:
        report_progress = Mock()
        cb = TrainingProgressCallback(report_progress, min_p=10.0, max_p=80.0)

        trainer = _make_trainer(max_epochs=2, num_training_batches=5)

        for i in range(3):
            cb.on_train_batch_end(
                trainer=trainer,
                pl_module=Mock(),
                outputs=Mock(),
                batch=Mock(),
                batch_idx=i,
            )

        assert cb._current_step == 3
        assert cb._total_steps == 10

        calls = report_progress.call_args_list
        assert len(calls) == 3

        # progress = min_p + (step / total) * (max_p - min_p)
        # steps: 1, 2, 3 (because we increment before emit)
        expected = [
            10.0 + (1 / 10) * 70.0,
            10.0 + (2 / 10) * 70.0,
            10.0 + (3 / 10) * 70.0,
        ]
        actual = [c.args[1] for c in calls]
        assert actual == expected

    def test_on_train_batch_end_respects_custom_progress_bounds(self) -> None:
        report_progress = Mock()
        cb = TrainingProgressCallback(report_progress, min_p=0.0, max_p=100.0)

        trainer = _make_trainer(max_epochs=1, num_training_batches=4)

        for i in range(4):
            cb.on_train_batch_end(
                trainer=trainer,
                pl_module=Mock(),
                outputs=Mock(),
                batch=Mock(),
                batch_idx=i,
            )

        expected = [25.0, 50.0, 75.0, 100.0]
        actual = [c.args[1] for c in report_progress.call_args_list]
        assert actual == expected
