# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class BaseIDModel(ABC, BaseModel):
    """Base model with an id field."""

    id: UUID = Field(default_factory=uuid4)


class BaseIDNameModel(ABC, BaseModel):
    """Base model with id and name fields."""

    id: UUID = Field(default_factory=uuid4)
    name: str = "Default Name"


class Pagination(ABC, BaseModel):
    """Pagination model."""

    offset: int  # index of the first item returned (0-based)
    limit: int  # number of items requested per page
    count: int  # number of items actually returned (may be less than limit if at the end)
    total: int  # total number of items available
