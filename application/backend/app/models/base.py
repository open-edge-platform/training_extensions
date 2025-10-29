# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, ConfigDict


class BaseEntity(BaseModel):
    model_config = ConfigDict(from_attributes=True)
