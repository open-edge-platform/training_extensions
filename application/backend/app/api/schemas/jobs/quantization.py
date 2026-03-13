# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.core.jobs.models import JobType
from app.models.jobs.quantization_job import QuantizationJob

from .base import BaseJobRequest


class QuantizationRequestParams(BaseModel):
    """Request parameters for the quantization job."""

    model_id: UUID = Field(..., description="ID of the model revision to quantize")
    device: str = Field(..., description="Device identifier for calibration (e.g., 'cpu', 'xpu-0')")
    max_calibration_subset_size: int = Field(
        100, description="Maximum number of samples from training set used for calibration"
    )
    max_drop: float | None = Field(
        None,
        description=(
            "Maximum allowed accuracy drop. If provided, uses nncf.quantize_with_accuracy_control(); "
            "if omitted, uses nncf.quantize()"
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_id": "6b7bb928-5d6f-46ea-8fd2-5ce80dd1e12b",
                "device": "cpu",
                "max_calibration_subset_size": 100,
                "max_drop": 0.01,
            }
        }
    }


class QuantizationRequest(BaseJobRequest):
    """Request schema for submitting a quantization job."""

    job_type: Literal[JobType.QUANTIZE]
    parameters: QuantizationRequestParams = Field(..., description="Parameters required for the quantization job")

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_type": "quantize",
                "project_id": "7b073838-99d3-42ff-9018-4e901eb047fc",
                "parameters": {
                    "model_id": "6b7bb928-5d6f-46ea-8fd2-5ce80dd1e12b",
                    "device": "cpu",
                    "max_calibration_subset_size": 100,
                    "max_drop": 0.01,
                },
            }
        }
    }


class QuantizationProjectMetadata(BaseModel):
    """Metadata about the project associated with a quantization job."""

    id: UUID = Field(..., description="Project identifier")


class QuantizationModelMetadata(BaseModel):
    """Metadata about the model being quantized."""

    id: UUID = Field(..., description="Model revision identifier")


class QuantizationModelVariantMetadata(BaseModel):
    """Metadata about the quantized model variant."""

    id: UUID = Field(..., description="Quantized model variant identifier")


class QuantizationMetadata(BaseModel):
    """Metadata associated with a quantization job."""

    project: QuantizationProjectMetadata = Field(..., description="Project associated with the quantization job")
    model: QuantizationModelMetadata = Field(..., description="Model being quantized")
    model_variant: QuantizationModelVariantMetadata = Field(..., description="Model variant created by quantization")
    device: str = Field(..., description="Device used for calibration")
    max_calibration_subset_size: int = Field(..., description="Maximum calibration subset size")
    max_drop: float | None = Field(None, description="Maximum allowed accuracy drop")

    @model_validator(mode="before")
    @classmethod
    def populate_metadata(cls, data: object) -> object:
        if isinstance(data, QuantizationJob):
            return {
                "project": QuantizationProjectMetadata(id=data.project_id),
                "model": QuantizationModelMetadata(id=data.params.model_id),
                "model_variant": QuantizationModelVariantMetadata(id=data.params.model_variant_id),
                "device": data.params.device.name,
                "max_calibration_subset_size": data.params.max_calibration_subset_size,
                "max_drop": data.params.max_drop,
            }
        return data
