# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter


class DataCollectionPolicyBase(BaseModel):
    type: str
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
