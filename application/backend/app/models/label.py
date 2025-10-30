# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from uuid import UUID

from pydantic import BaseModel

from .base import BaseEntity


class LabelReference(BaseModel):
    id: UUID
    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
            }
        }
    }


class Label(BaseEntity):
    id: UUID
    project_id: UUID
    name: str
    color: str
    hotkey: str | None
