# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from uuid import UUID

from pydantic import model_validator

from app.core.jobs.models import JobParams, JobType, ProjectJob
from app.models import DatasetFormat, DatasetItemSubset
from app.models.project import Task
from app.models.task import TaskType

VALID_FORMATS_PER_TASK = {
    TaskType.CLASSIFICATION: [DatasetFormat.GETI, DatasetFormat.VOC],
    TaskType.DETECTION: [DatasetFormat.GETI, DatasetFormat.COCO, DatasetFormat.YOLO],
    TaskType.INSTANCE_SEGMENTATION: [DatasetFormat.GETI, DatasetFormat.COCO],
}


class ExportDatasetJobParams(JobParams):
    dataset_id: UUID | None = None
    project_id: UUID
    task: Task
    export_format: DatasetFormat
    labels: list[str] | None = None
    subsets: list[DatasetItemSubset] | None = None
    include_unannotated: bool = True

    @model_validator(mode="after")
    def validate_format(self) -> "ExportDatasetJobParams":
        """Validate that the export format is compatible with the task type."""
        if (
            self.task.task_type in VALID_FORMATS_PER_TASK
            and self.export_format not in VALID_FORMATS_PER_TASK[self.task.task_type]
        ):
            allowed = ", ".join(f.value for f in VALID_FORMATS_PER_TASK[self.task.task_type])
            raise ValueError(
                f"Export format '{self.export_format}' is not supported for {self.task.task_type} task. "
                f"Allowed formats are: {allowed}"
            )
        return self


class ExportDatasetJob(ProjectJob[ExportDatasetJobParams]):
    job_type: JobType = JobType.EXPORT_DATASET  # pyrefly: ignore[bad-override]
    params: ExportDatasetJobParams
