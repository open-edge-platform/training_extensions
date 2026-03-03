# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from app.core.jobs.models import JobParams
from app.execution.base import Execution, step


class TestStepDecorator:
    """Test suite for the @step decorator."""

    @pytest.mark.parametrize("percent", [0, 50])
    def test_step_calls_report_progress_on_success(self, percent):
        """Test that the decorator calls report_progress when the step starts and completes."""

        # Arrange
        class MockTrainer(Execution[JobParams]):
            def execute(self, params: JobParams) -> None:
                pass

            @step("Test Step", percent)
            def test_method(self) -> str:
                return "result"

        trainer = MockTrainer()
        with (
            patch.object(trainer, "update_message") as mock_update_message,
            patch.object(trainer, "_report_progress") as mock_report_progress,
        ):
            # Act
            result = trainer.test_method()

            # Assert
            assert result == "result"
            mock_update_message.assert_called_once_with("Started: Test Step")
            mock_report_progress.assert_called_once_with("Completed: Test Step", percent=percent)

    def test_step_decorator_reports_failure_on_exception(self):
        """Test that the decorator reports failure when an exception occurs."""

        # Arrange
        class CustomException(Exception):
            pass

        class MockTrainer(Execution[JobParams]):
            def execute(self, params: JobParams) -> None:
                pass

            @step("Failing Step")
            def failing_method(self) -> None:
                raise CustomException("Custom error")

        trainer = MockTrainer()

        with (
            pytest.raises(CustomException, match="Custom error"),
            patch.object(trainer, "update_message") as mock_update_message,
            patch.object(trainer, "update_message_with_stacktrace") as mock_update_message_with_stacktrace,
        ):
            # Act
            trainer.failing_method()

            # Assert
            mock_update_message.assert_called_once_with("Failed: Test error")
            mock_update_message_with_stacktrace.assert_called_once_with("Failed: Failing Step")
