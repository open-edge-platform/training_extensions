# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.api.schemas.jobs.base import BaseJobRequest
from app.core.jobs.models import JobType
from app.models import DatasetItemSubset, TaskType


class BaseImportDatasetRequest(BaseModel):
    staged_dataset_id: UUID = Field(..., description="ID of the staged dataset associated with the job")


class PrepareImportDatasetRequest(BaseImportDatasetRequest):
    job_type: Literal[JobType.IMPORT_DATASET_PREPARE]

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_type": "train",
                "staged_dataset_id": "7b073838-99d3-42ff-9018-4e901eb047fc",
            }
        }
    }


class DatasetFilters(BaseModel):
    labels: list[str] = Field(..., description="List of labels associated with the dataset")
    subsets: list[str] = Field(..., description="List of subsets associated with the dataset")
    include_unannotated: bool = Field(..., description="Whether to include unannotated items to the dataset")

    model_config = {
        "json_schema_extra": {
            "example": {
                "labels": ["person", "car", "motorcycle"],
                "subsets": ["training", "validation"],
                "include_unannotated": False,
            }
        }
    }


class ImportDatasetProjectParams(BaseModel):
    filters: DatasetFilters = Field(..., description="Filters to apply to the dataset during import")
    labels_mapping: dict[str, str] = Field(..., description="Mapping between labels")
    subset_mapping: dict[str, DatasetItemSubset] = Field(
        ..., description="Subset mapping between dataset and project convention"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "filters": {
                    "labels": ["person", "car", "motorcycle"],
                    "subsets": ["training", "validation"],
                    "include_unannotated": False,
                },
                "labels_mapping": {"car": "vehicle", "motorcycle": "vehicle", "person": "person"},
                "subset_mapping": {"train": "training", "validation": "validation", "unassigned": "testing"},
            }
        }
    }


class ImportDatasetProjectRequest(BaseImportDatasetRequest, BaseJobRequest):
    job_type: Literal[JobType.IMPORT_DATASET_PROJECT]

    parameters: ImportDatasetProjectParams = Field(..., description="Dataset parameters")

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_type": "import_dataset_to_project",
                "staged_dataset_id": "63f983fe-f2c7-4054-a0b1-6aab8a355a12",
                "project_id": "103b9b76-ada6-4381-91bf-fa315fe5cb66",
                "parameters": {
                    "filters": {
                        "labels": ["person", "car", "motorcycle"],
                        "subsets": ["training", "validation"],
                        "include_unannotated": False,
                    },
                    "labels_mapping": {"car": "vehicle", "motorcycle": "vehicle", "person": "person"},
                    "subset_mapping": {"train": "training", "validation": "validation", "unassigned": "testing"},
                },
            }
        }
    }


class NewProjectParams(BaseModel):
    name: str = Field(..., description="New project parameter name")
    task_type: TaskType = Field(..., description="New project task type")
    labels: list[str] = Field(..., description="New project labels")

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "New Project from Imported Dataset",
                "task_type": "object_detection",
                "labels": ["person", "vehicle"],
            }
        }
    }


class ImportDatasetNewParams(BaseModel):
    project: NewProjectParams = Field(..., description="New project parameters")
    filters: DatasetFilters = Field(..., description="Filters to apply to the dataset during import")

    model_config = {
        "json_schema_extra": {
            "example": {
                "project": {
                    "name": "New Project from Imported Dataset",
                    "task_type": "object_detection",
                    "labels": ["person", "vehicle"],
                },
                "filters": {
                    "labels": ["person", "car", "motorcycle"],
                    "subsets": ["train", "validation"],
                    "include_unannotated": False,
                },
            }
        }
    }


class ImportDatasetNewRequest(BaseImportDatasetRequest):
    job_type: Literal[JobType.IMPORT_DATASET_NEW]

    parameters: ImportDatasetNewParams = Field(..., description="Dataset parameters")

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_type": "import_dataset_as_new_project",
                "staged_dataset_id": "63f983fe-f2c7-4054-a0b1-6aab8a355a12",
                "parameters": {
                    "project": {
                        "name": "New Project from Imported Dataset",
                        "task_type": "object_detection",
                        "labels": ["person", "vehicle"],
                    },
                    "filters": {
                        "labels": ["person", "car", "motorcycle"],
                        "subsets": ["train", "validation"],
                        "include_unannotated": False,
                    },
                },
            }
        }
    }
