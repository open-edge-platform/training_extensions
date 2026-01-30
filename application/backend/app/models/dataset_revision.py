# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from uuid import UUID

from app.models.base import BaseEntity


class DatasetRevision(BaseEntity):
    """
    A dataset revision is an immutable snapshot of a training dataset.

    Attributes:
        id: Unique identifier for the dataset revision.
        name: Name of the dataset revision.
        created_at: Timestamp indicating when the dataset revision was created.
        files_deleted: Flag indicating whether the files associated with this dataset revision have been deleted.
    """

    id: UUID
    name: str
    created_at: datetime
    files_deleted: bool
