# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from app.models.base import BaseEntity

from .task import Task


class Project(BaseEntity):
    """
    Represents a machine learning project.

    A project is the top-level container that organizes tasks, pipelines, datasets, and models for a specific machine
    learning workflow.

    Attributes:
        id: Unique identifier for the project.
        active_pipeline: Flag indicating whether the project has an active running pipeline. Defaults to False.
        name: Human-readable name of the project.
        task: The annotation task configuration associated with this project.
    """

    id: UUID
    active_pipeline: bool = False
    name: str
    task: Task
