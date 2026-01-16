# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter


class DataCollectionPolicyBase(BaseModel):
    enabled: bool = True


class FixedRateDataCollectionPolicy(DataCollectionPolicyBase):
    type: Literal["fixed_rate"] = "fixed_rate"
    rate: float


class ConfidenceThresholdDataCollectionPolicy(DataCollectionPolicyBase):
    type: Literal["confidence_threshold"] = "confidence_threshold"
    confidence_threshold: float
    min_sampling_interval: float


DataCollectionPolicy = Annotated[
    FixedRateDataCollectionPolicy | ConfidenceThresholdDataCollectionPolicy, Field(discriminator="type")
]

DataCollectionPolicyAdapter: TypeAdapter[DataCollectionPolicy] = TypeAdapter(DataCollectionPolicy)


class DataCollectionConfig(BaseModel):
    """
    Configuration for data collection during pipeline execution.

    Attributes:
        max_dataset_size: Maximum number of items allowed in the dataset. When reached,
            data collection will be disabled to prevent uncontrolled dataset growth.
            Set to None for unlimited collection (default).
        policies: List of policies governing data collection behavior.
    """

    max_dataset_size: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of items allowed in the dataset. None for unlimited.",
    )
    policies: list[DataCollectionPolicy] = Field(default_factory=list)
