# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, call

import pytest

from app.core.run import ExecutionContext
from app.services.training.base import Trainer, step


class TestStepDecorator:
    """Test suite for the @step decorator."""

    @pytest.mark.parametrize("percent", [None, 50])
    def test_step_calls_report_progress_on_success(self, percent):
        """Test that the decorator calls report_progress when the step starts and completes."""

        # Arrange
        class MockTrainer(Trainer):
            def run(self, ctx: ExecutionContext) -> None:
                pass

            @step("Test Step", percent)
            def test_method(self) -> str:
                return "result"

        trainer = MockTrainer()
        trainer.report_progress = Mock()

        # Act
        result = trainer.test_method()

        # Assert
        assert result == "result"
        assert trainer.report_progress.call_args_list == [
            call("Started: Test Step"),
            call("Completed: Test Step", percent=percent),
        ]

    def test_step_decorator_reports_failure_on_exception(self):
        """Test that the decorator reports failure when an exception occurs."""

        # Arrange
        class MockTrainer(Trainer):
            def run(self, ctx: ExecutionContext) -> None:
                pass

            @step("Failing Step")
            def failing_method(self) -> None:
                raise ValueError("Test error")

        trainer = MockTrainer()
        trainer.report_progress = Mock()

        # Act & Assert
        with pytest.raises(ValueError, match="Test error"):
            trainer.failing_method()

        assert trainer.report_progress.call_args_list == [
            call("Started: Failing Step"),
            call("Failed: Failing Step", exc=True),
        ]

    def test_step_decorator_preserves_exception(self):
        """Test that the decorator re-raises the original exception."""

        # Arrange
        class CustomException(Exception):
            pass

        class MockTrainer(Trainer):
            def run(self, ctx: ExecutionContext) -> None:
                pass

            @step("Exception Step")
            def exception_method(self) -> None:
                raise CustomException("Custom error")

        trainer = MockTrainer()
        trainer.report_progress = Mock()  # type: ignore[method-assign]

        # Act & Assert
        with pytest.raises(CustomException, match="Custom error"):
            trainer.exception_method()
