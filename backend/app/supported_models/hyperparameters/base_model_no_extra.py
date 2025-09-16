# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel, ConfigDict


class BaseModelNoExtra(BaseModel):
    """Base model class that do not allow for extra fields."""

    model_config = ConfigDict(extra="forbid")
