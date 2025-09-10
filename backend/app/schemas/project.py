# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from pydantic import BaseModel, field_serializer, field_validator

from app.schemas.base import BaseIDNameModel


class Label(BaseModel):
    name: str


class Task(BaseModel):
    task_type: Literal["classification", "detection", "segmentation"]
    exclusive_labels: bool = False
    labels: list[Label] = []

    @field_serializer("labels", when_used="json")
    def serialize_labels(self, labels: list[Label]) -> list[str]:
        return [label.name for label in labels]

    @field_validator("labels", mode="before")
    def parse_labels(cls, v: list[Any]) -> list[Label]:
        if v and all(isinstance(v, str) for v in v):
            return [Label(name=item) for item in v]
        return v


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
                    "labels": [{"name": "cat"}, {"name": "dog"}],
                },
            }
        }
    }
