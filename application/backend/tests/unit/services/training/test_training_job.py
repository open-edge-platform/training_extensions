# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch
from uuid import uuid4

import pytest

from app.models import TaskType
from app.schemas.project import TaskBase
from app.services.training.models import TrainingJob, TrainingParams


@pytest.fixture
def fxt_training_params():
    return TrainingParams(model_architecture_id="test_arch", task=TaskBase(task_type=TaskType.CLASSIFICATION))


@pytest.fixture
def fxt_training_job(tmp_path, fxt_training_params):
    project_id = uuid4()
    log_dir = tmp_path / "logs"
    data_dir = tmp_path / "data"
    log_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    return TrainingJob(
        id=uuid4(),
        project_id=project_id,
        log_dir=log_dir,
        data_dir=data_dir,
        params=fxt_training_params,
    )


class TestTrainingJob:
    def test_on_finish_copies_log_file(self, fxt_training_job):
        """Test that log file is copied to the correct destination."""
        # Create the log file
        log_path = fxt_training_job.log_dir / fxt_training_job.log_file
        log_path.write_text("Training log content")

        # Execute
        fxt_training_job.on_finish()

        # Verify the log was copied
        expected_path = (
            fxt_training_job.data_dir
            / "projects"
            / str(fxt_training_job.project_id)
            / "models"
            / str(fxt_training_job.params.model_id)
            / "training.log"
        )
        assert expected_path.exists()
        assert expected_path.read_text() == "Training log content"

    @patch("app.services.training.models.logger")
    def test_on_finish_logs_warning(self, mock_logger, fxt_training_job):
        """Test that a warning is logged and no file copied when the source log file doesn't exist."""
        # Don't create the log file

        # Execute
        fxt_training_job.on_finish()

        # Verify warning was logged and no file was copied
        log_path = fxt_training_job.log_dir / fxt_training_job.log_file
        mock_logger.warning.assert_called_once_with(f"Log file {log_path} does not exist")
        expected_path = (
            fxt_training_job.data_dir
            / "projects"
            / str(fxt_training_job.project_id)
            / "models"
            / str(fxt_training_job.params.model_id)
            / "training.log"
        )
        assert not expected_path.exists()
