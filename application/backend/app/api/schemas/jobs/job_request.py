# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import Field, TypeAdapter

from .dataset_import import ImportDatasetNewRequest, ImportDatasetProjectRequest, PrepareImportDatasetRequest
from .training import TrainingRequest

JobRequest = Annotated[
    TrainingRequest | ImportDatasetProjectRequest | PrepareImportDatasetRequest | ImportDatasetNewRequest,
    Field(discriminator="job_type"),
]

JobRequestAdapter: TypeAdapter[JobRequest] = TypeAdapter(JobRequest)
