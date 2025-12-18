# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from app.models import DatasetItemFormat, DatasetItemSubset
from app.models.base import BaseEntity


class DatasetRevisionItem(BaseEntity):
    """
    DatasetRevisionItem represents an individual item within a dataset revision.

    Attributes:
        id: Unique identifier for the dataset revision item.
        name: Name of the dataset revision item.
        format: Format of the dataset revision item (e.g., JPG, PNG).
        width: Width of the dataset revision item in pixels.
        height: Height of the dataset revision item in pixels.
        subset: Subset to which the dataset revision item belongs (e.g., training, validation, testing).
    """

    id: UUID
    name: str
    format: DatasetItemFormat
    width: int
    height: int
    subset: DatasetItemSubset
