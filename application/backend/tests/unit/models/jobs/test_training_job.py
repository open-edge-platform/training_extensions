# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from app.models import Task, TaskType, TrainingJob, TrainingJobParams
from app.models.system import DeviceInfo, DeviceType


@pytest.fixture
def fxt_training_params() -> Callable[[UUID, UUID], TrainingJobParams]:
    def _make_training_job_params(job_id: UUID, project_id: UUID) -> TrainingJobParams:
        return TrainingJobParams(
            device=DeviceInfo(type=DeviceType.XPU, name="Intel Arc B580", memory=12884901888, index=0),
            model_architecture_id="test_arch",
            model_architecture_name="Test Arch",
            task=Task(task_type=TaskType.CLASSIFICATION),
            job_id=job_id,
            project_id=project_id,
        )

    return _make_training_job_params


@pytest.fixture
def fxt_training_job(tmp_path, fxt_training_params):
    job_id = uuid4()
    project_id = uuid4()
    log_dir = tmp_path / "logs"
    data_dir = tmp_path / "data"
    log_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    return TrainingJob(
        id=job_id,
        project_id=project_id,
        log_dir=log_dir,
        data_dir=data_dir,
        params=fxt_training_params(job_id, project_id),
    )


class TestTrainingJob:
    def test_on_complete_copies_log_file(self, fxt_training_job):
        """Test that log file is copied to the correct destination."""
        # Create the log file
        log_path = fxt_training_job.log_dir / fxt_training_job.log_file
        log_path.write_text("Training log content")

        # Execute
        fxt_training_job.on_complete()

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

    @patch("app.models.jobs.training_job.logger")
    def test_on_complete_logs_warning(self, mock_logger, fxt_training_job):
        """Test that a warning is logged and no file copied when the source log file doesn't exist."""
        # Don't create the log file

        # Execute
        fxt_training_job.on_complete()

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

    def test_on_complete_removes_getitune_workspace(self, fxt_training_job):
        """Test that the getitune workspace directory is removed on job completion."""
        # Create the workspace directory with a timestamped subdir and a stray file
        workspace_dir = fxt_training_job.data_dir / f"getitune-workspace-{fxt_training_job.params.model_id}"
        timestamp_dir = workspace_dir / "20260101_000000"
        timestamp_dir.mkdir(parents=True)
        (timestamp_dir / "leftover.txt").write_text("temp")

        fxt_training_job.on_complete()

        assert not workspace_dir.exists()

    def test_on_complete_no_op_when_workspace_missing(self, fxt_training_job):
        """on_complete must not raise when the getitune workspace was never created."""
        workspace_dir = fxt_training_job.data_dir / f"getitune-workspace-{fxt_training_job.params.model_id}"
        assert not workspace_dir.exists()

        # Should not raise
        fxt_training_job.on_complete()
