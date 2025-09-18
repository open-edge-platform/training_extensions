# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ModelActivationState(BaseModel):
    active_model_id: UUID | None = Field(..., description="ID of the model that is currently used for inference")
    available_models: list[UUID] = Field(..., description="List of all available model IDs that can be activated")

    @field_validator("active_model_id")
    @classmethod
    def validate_active_model(cls, v, info):  # noqa: ANN001
        if v is not None and "available_models" in info.data and v not in info.data["available_models"]:
            raise ValueError(
                f"active_model_id '{v}' must be one of the available_models: {info.data['available_models']}"
            )
        return v

    def to_json_dict(self) -> dict:
        """Serialize the state to a JSON-compatible dictionary."""
        return self.model_dump()

    @classmethod
    def from_json_dict(cls, data: dict) -> "ModelActivationState":
        """Deserialize the state from a JSON-compatible dictionary."""
        return cls.model_validate(data)
