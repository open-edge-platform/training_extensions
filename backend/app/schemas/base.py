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
