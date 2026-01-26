# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field

from app.core.models import BaseRequiredIDModel


class StagedDatasetView(BaseRequiredIDModel):
    format: str = Field(..., description="Dataset format, e.g., 'coco', 'datumaro'")
    compressed: bool = Field(..., description="Whether the dataset is compressed")
    ready_for_export: bool = Field(..., description="Whether the dataset is ready for export")
    ready_for_import: bool = Field(..., description="Whether the dataset is ready for import")
    size: int = Field(..., description="Dataset size in bytes")
    metadata: dict | None = Field(None, description="Dataset metadata")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "63f983fe-f2c7-4054-a0b1-6aab8a355a12",
                "format": "datumaro",
                "compressed": False,
                "ready_for_export": False,
                "ready_for_import": True,
                "size": 987654321,
                "metadata": {
                    "num_items": 2000,
                    "annotation_type": "bounding_box",
                    "num_annotations": 10000,
                    "labels": ["person", "bicycle", "tree"],
                },
            },
        }
    }
