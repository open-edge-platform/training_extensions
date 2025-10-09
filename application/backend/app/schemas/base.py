# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class HasID(ABC, BaseModel):
    """Mixin: Optional UUID with auto-generation."""

    id: UUID = Field(default_factory=uuid4, description="Unique identifier")


class RequiresID(ABC, BaseModel):
    """Mixin: Required UUID field."""

    id: UUID = Field(..., description="Unique identifier")


class HasName(ABC, BaseModel):
    """Mixin: Optional name with default value."""

    name: str = Field(default="Default Name", description="Name of the entity")


class RequiresName(ABC, BaseModel):
    """Mixin: Required name field."""

    name: str = Field(..., description="Name of the entity")


class BaseIDModel(HasID):
    """Base model with auto-generated ID."""


class BaseIDNameModel(HasID, HasName):
    """Base model with auto-generated ID and default name."""


class BaseRequiredIDModel(RequiresID):
    """Base model with required ID."""


class BaseRequiredIDNameModel(RequiresID, RequiresName):
    """Base model with required ID and name."""


class Pagination(ABC, BaseModel):
    """Pagination model."""

    offset: int  # index of the first item returned (0-based)
    limit: int  # number of items requested per page
    count: int  # number of items actually returned (may be less than limit if at the end)
    total: int  # total number of items available
