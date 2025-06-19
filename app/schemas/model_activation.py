from pydantic import BaseModel, Field, field_validator


class ModelActivationState(BaseModel):
    active_model: str | None = Field(..., description="Name of the model that is currently used for inference")
    available_models: list[str] = Field(..., description="List of all available model names that can be activated")

    @field_validator("active_model")
    @classmethod
    def validate_active_model(cls, v, info):  # noqa: ANN001
        if v is not None and "available_models" in info.data and v not in info.data["available_models"]:
            raise ValueError(f"active_model '{v}' must be one of the available_models: {info.data['available_models']}")
        return v

    def to_json_dict(self) -> dict:
        """Serialize the state to a JSON-compatible dictionary."""
        return self.model_dump()

    @classmethod
    def from_json_dict(cls, data: dict) -> "ModelActivationState":
        """Deserialize the state from a JSON-compatible dictionary."""
        return cls.model_validate(data)
