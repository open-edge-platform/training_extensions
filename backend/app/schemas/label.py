# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field, StringConstraints

from app.schemas.base import BaseIDModel

COLOR_REGEX = r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$"


class LabelToEdit(BaseModel):
    id: UUID = Field(..., description="UUID of the label to edit")
    new_name: str | None = Field(None, min_length=1, max_length=50, description="New label name")
    new_color: Annotated[str | None, StringConstraints(pattern=COLOR_REGEX)] = Field(
        None, description="New hex color code, e.g. #RRGGBB or #RGB"
    )
    new_hotkey: str | None = Field(None, description="New hotkey")

    def to_label(self) -> "Label":
        return Label(
            id=self.id,
            name=self.new_name,
            color=self.new_color,
            hotkey=self.new_hotkey,
        )

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "new_name": "Updated Label",
                "new_color": "#FFAABB",
                "new_hotkey": "A",
            }
        }
    }


class LabelToAdd(BaseModel):
    name: str = Field(..., min_length=1, max_length=50, description="Label name")
    color: Annotated[str | None, StringConstraints(pattern=COLOR_REGEX)] = Field(
        ..., description="Hex color code, e.g. #RRGGBB or #RGB"
    )
    hotkey: str | None = Field(None, description="Hotkey to assign")

    def to_label(self) -> "Label":
        return Label(
            name=self.name,
            color=self.color,
            hotkey=self.hotkey,
        )

    model_config = {"json_schema_extra": {"example": {"name": "New Label", "color": "#ABC123", "hotkey": "N"}}}


class LabelToRemove(BaseModel):
    id: UUID = Field(..., description="UUID of the label to remove")

    model_config = {"json_schema_extra": {"example": {"id": "123e4567-e89b-12d3-a456-426614174000"}}}


class PatchLabels(BaseModel):
    labels_to_edit: list[LabelToEdit] = Field(default_factory=list, description="List of labels to edit")
    labels_to_add: list[LabelToAdd] = Field(default_factory=list, description="List of labels to add")
    labels_to_remove: list[LabelToRemove] = Field(default_factory=list, description="List of labels to remove")

    model_config = {
        "json_schema_extra": {
            "example": {
                "labels_to_edit": [
                    {
                        "id": "123e4567-e89b-12d3-a456-426614174000",
                        "new_name": "Updated Label",
                        "new_color": "#FFAABB",
                        "new_hotkey": "A",
                    }
                ],
                "labels_to_add": [{"name": "New Label", "color": "#ABC123", "hotkey": "N"}],
                "labels_to_remove": [{"id": "123e4567-e89b-12d3-a456-426614174000"}],
            }
        }
    }


class Label(BaseIDModel):
    name: str | None = Field(None, min_length=1, max_length=50, description="Label name")
    color: Annotated[str | None, StringConstraints(pattern=COLOR_REGEX)] = Field(
        None, description="Hex color code, e.g. #RRGGBB or #RGB"
    )
    hotkey: str | None = Field(None, description="Hotkey assigned to the label")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "Sample Label",
                "color": "#FF5733",
                "hotkey": "S",
            }
        }
    }
