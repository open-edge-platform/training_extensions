# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from uuid import UUID

from app.models.base import BaseEntity


class DatasetRevision(BaseEntity):
    """
    Represents a specific revision of a dataset.

    A dataset revision captures the state of a dataset at a particular point in time, allowing for versioning and
    tracking changes over time.

    Attributes:
        id: Unique identifier for the dataset revision.
        project_id: Identifier of the project to which this dataset revision belongs.
        files_deleted: Flag indicating whether the files associated with this dataset revision have been deleted.
    """

    id: UUID
    project_id: UUID
    files_deleted: bool
