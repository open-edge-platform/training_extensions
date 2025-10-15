# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

from app.schemas.base import HasID, RequiresID
from app.schemas.label import LabelCreate, LabelView


class TaskType(StrEnum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"


class TaskBase(BaseModel):
    task_type: TaskType = Field(description="Task type (classification, detection or segmentation).")
    exclusive_labels: bool = Field(
        default=False, description="Whether labels are exclusive (e.g. classification) or not (e.g. detection)."
    )


class TaskView(TaskBase):
    labels: list[LabelView] = Field(default_factory=list, description="List of task labels.")


class TaskCreate(TaskBase):
    labels: list[LabelCreate] = Field(default_factory=list, description="List of task labels to create.")


class ProjectUpdateName(BaseModel):
    """Schema for updating project name"""

    name: str

    model_config = {"json_schema_extra": {"example": {"name": "updated_project_name"}}}


T = TypeVar("T", bound=TaskBase)  # Task type e.g. TaskView, TaskCreate


class ProjectBase(BaseModel, Generic[T]):
    name: str = Field(..., description="Name of the project.")
    task: T

    @classmethod
    def example_dict(cls, view: bool = False) -> dict:
        """Generate example with configurable ID inclusion."""
        labels = [
            {
                **({"id": "a22d82ba-afa9-4d6e-bbc1-8c8e4002ec29", "color": "#FF5733", "hotkey": "S"} if view else {}),
                "name": "cat",
            },
            {
                **({"id": "8aa85368-11ba-4507-88f2-6a6704d78ef5", "color": "#33FF57", "hotkey": "D"} if view else {}),
                "name": "dog",
            },
        ]

        return {
            **({"id": "7b073838-99d3-42ff-9018-4e901eb047fc"} if view else {}),
            "name": "animals",
            "active_pipeline": False if view else None,
            "task": {
                "task_type": "classification",
                "exclusive_labels": True,
                "labels": labels,
            },
        }


class ProjectCreate(HasID, ProjectBase[TaskCreate]):
    model_config = {
        "json_schema_extra": {
            "example": ProjectBase.example_dict(),
        }
    }


class ProjectView(RequiresID, ProjectBase[TaskView]):
    active_pipeline: bool = Field(..., description="Whether the project has an active pipeline.")

    model_config = {
        "json_schema_extra": {
            "example": ProjectBase.example_dict(view=True),
        }
    }
