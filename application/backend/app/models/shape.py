# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class Point(BaseModel):
    x: int = Field(..., description="Point x coordinate", ge=0)
    y: int = Field(..., description="Point y coordinate", ge=0)


# Base class with discriminator field
class ShapeBase(BaseModel):
    type: str


class Rectangle(ShapeBase):
    type: Literal["rectangle"] = "rectangle"
    x: int = Field(..., description="Rectangle x coordinate", ge=0)
    y: int = Field(..., description="Rectangle y coordinate", ge=0)
    width: int = Field(..., description="Rectangle width", ge=0)
    height: int = Field(..., description="Rectangle height", ge=0)

    model_config = {
        "json_schema_extra": {"example": {"type": "rectangle", "x": 10, "y": 20, "width": 100, "height": 200}}
    }


class Polygon(ShapeBase):
    type: Literal["polygon"] = "polygon"
    points: list[Point] = Field(..., description="Polygon points")

    model_config = {"json_schema_extra": {"example": {"type": "polygon", "points": [[10, 20], [20, 60], [30, 40]]}}}


class FullImage(ShapeBase):
    type: Literal["full_image"] = "full_image"

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "full_image",
            }
        }
    }


Shape = Annotated[Rectangle | Polygon | FullImage, Field(discriminator="type")]
