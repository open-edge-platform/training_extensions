# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""License schemas"""

from pydantic import BaseModel, Field


class LicenseAcceptResponse(BaseModel):
    """Response schema for the license accept endpoint."""

    license_accepted: bool = Field(..., description="Whether the license has been accepted")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"license_accepted": True},
            ]
        }
    }
