# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from enum import StrEnum

from pydantic import BaseModel

from app.schemas.base import BaseIDNameModel
from app.schemas.label import Label


class TaskType(StrEnum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"


class Task(BaseModel):
    task_type: TaskType
    exclusive_labels: bool = False
    labels: list[Label] = []


class Project(BaseIDNameModel):
    task: Task

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "7b073838-99d3-42ff-9018-4e901eb047fc",
                "name": "animals",
                "task": {
                    "task_type": "classification",
                    "exclusive_labels": True,
                    "labels": [
                        {
                            "id": "a22d82ba-afa9-4d6e-bbc1-8c8e4002ec29",
                            "name": "cat",
                            "color": "#FF5733",
                            "hotkey": "S",
                        },
                        {
                            "id": "8aa85368-11ba-4507-88f2-6a6704d78ef5",
                            "name": "dog",
                            "color": "#33FF57",
                            "hotkey": "D",
                        },
                    ],
                },
            }
        }
    }
