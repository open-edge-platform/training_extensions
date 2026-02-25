# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.api.schemas.dataset import DatasetFilters
from app.core.jobs.models import JobType
from app.models import DatasetItemSubset, TaskType
from app.models.jobs import ImportDatasetToProjectJob, PrepareDatasetForImportJob

from .base import BaseJobRequest


class BaseImportRequest(BaseModel):
    staged_dataset_id: UUID = Field(..., description="ID of the staged dataset to import")


class PrepareDatasetForImportRequest(BaseImportRequest):
    job_type: Literal[JobType.PREPARE_DATASET_FOR_IMPORT]

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_type": "prepare_dataset_for_import",
                "staged_dataset_id": "7b073838-99d3-42ff-9018-4e901eb047fc",
            }
        }
    }


class ImportDatasetProjectParams(BaseModel):
    labels_mapping: dict[str, str] | None = Field(
        None,
        description="Specify how to map the labels found in the dataset to the labels defined in the project. If and "
        "only if the dataset labels exactly match the project labels, this parameter can be left unspecified (null)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "labels_mapping": {"car": "vehicle", "motorcycle": "vehicle", "person": "person"},
            }
        }
    }


class ImportDatasetToProjectRequest(BaseImportRequest, BaseJobRequest):
    job_type: Literal[JobType.IMPORT_DATASET_TO_PROJECT]

    parameters: ImportDatasetProjectParams = Field(
        default_factory=ImportDatasetProjectParams,
        description="Configuration for importing the dataset into the project, including filtering, labeling and "
        "subset mapping options",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_type": "import_dataset_to_project",
                "staged_dataset_id": "63f983fe-f2c7-4054-a0b1-6aab8a355a12",
                "project_id": "103b9b76-ada6-4381-91bf-fa315fe5cb66",
                "parameters": {
                    "labels_mapping": {"car": "vehicle", "motorcycle": "vehicle", "person": "person"},
                },
            }
        }
    }


class NewProjectParams(BaseModel):
    name: str = Field(..., description="Name to assign to the new project")
    task_type: TaskType = Field(..., description="Type of the project to create")
    labels: list[str] = Field(..., description="Labels to create in the new project")
    exclusive_labels: bool = Field(
        False,
        description="For classification projects: If True, multiple labels per item are allowed (multi-label); "
        "if False, exactly one label per item is allowed (multi-class)",
    )

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
    filters: DatasetFilters = Field(
        default_factory=DatasetFilters, description="Filters to apply to the dataset during import"
    )

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


class ImportDatasetAsNewProjectRequest(BaseImportRequest):
    job_type: Literal[JobType.IMPORT_DATASET_AS_NEW_PROJECT]

    parameters: ImportDatasetNewParams = Field(
        ...,
        description="Configuration for importing the dataset as the new project, including filtering options and new "
        "project parameters",
    )

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


class ImportDatasetMetadata(BaseModel):
    staged_dataset_id: UUID = Field(..., description="Dataset ID")
    project_id: UUID | None = Field(None, description="Project ID")
    filters: DatasetFilters | None = Field(None, description="Filters to apply to the dataset during import")
    labels_mapping: dict[str, str] | None = Field(
        None,
        description="Specify how to map the labels found in the dataset to the labels defined in the project. If and "
        "only if the dataset labels exactly match the project labels, this parameter can be left unspecified (null)",
    )
    subset_mapping: dict[str, DatasetItemSubset] | None = Field(
        None,
        description="Specify how to map the subsets assigned to the items in the dataset to the project subsets. "
        "If this parameter is unspecified (null), then each item will be assigned to the respective subset defined in "
        "the dataset",
    )
    project: NewProjectParams | None = Field(None, description="New project parameters")

    @model_validator(mode="before")
    @classmethod
    def populate_metadata(cls, data: object) -> object:
        if isinstance(data, PrepareDatasetForImportJob):
            return {
                "staged_dataset_id": data.params.staged_dataset_id,
            }
        if isinstance(data, ImportDatasetToProjectJob):
            return {
                "staged_dataset_id": data.params.staged_dataset_id,
                "project_id": data.params.project_id,
                "labels_mapping": data.params.labels_mapping,
            }

        return data
