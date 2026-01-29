# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import Field, TypeAdapter

from .dataset_export import ExportDatasetRequest, StageDatasetRequest
from .dataset_import import (
    ImportDatasetAsNewProjectRequest,
    ImportDatasetToProjectRequest,
    PrepareDatasetForImportRequest,
)
from .training import TrainingRequest

JobRequest = Annotated[
    TrainingRequest
    | ImportDatasetToProjectRequest
    | PrepareDatasetForImportRequest
    | ImportDatasetAsNewProjectRequest
    | ExportDatasetRequest
    | StageDatasetRequest,
    Field(discriminator="job_type"),
]

JobRequestAdapter: TypeAdapter[JobRequest] = TypeAdapter(JobRequest)
