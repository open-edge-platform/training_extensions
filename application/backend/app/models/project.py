# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from pydantic import Field

from app.models.base import BaseEntity

from .label import Label
from .task_type import TaskType


class Task(BaseEntity):
    exclusive_labels: bool = False
    task_type: TaskType
    labels: list[Label] = Field(default_factory=list)


class Project(BaseEntity):
    id: UUID
    active_pipeline: bool = False
    name: str
    task: Task
