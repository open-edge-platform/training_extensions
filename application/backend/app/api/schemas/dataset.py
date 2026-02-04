# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field, model_validator

from app.core.models import BaseRequiredIDModel
from app.models import DatasetFormat, StagedDataset


class DatasetFilters(BaseModel):
    labels: list[str] | None = Field(
        None,
        description="List of labels to consider during import or export; any annotation with labels not present in "
        "the list will be filtered out; if the parameter is unspecified (null), then all labels will be considered",
    )
    subsets: list[str] | None = Field(
        None,
        description="List of subsets to consider during import or export; any item assigned a subset not present in "
        "the list will be filtered out; if the parameter is unspecified (null), then all subsets will be considered",
    )
    include_unannotated: bool = Field(True, description="Whether to include unannotated items to the dataset")

    model_config = {
        "json_schema_extra": {
            "example": {
                "labels": ["person", "car", "motorcycle"],
                "subsets": ["training", "validation"],
                "include_unannotated": False,
            }
        }
    }


class StagedDatasetView(BaseRequiredIDModel):
    format: DatasetFormat = Field(..., description="Dataset format")
    compressed: bool = Field(..., description="Whether the dataset is compressed")
    ready_for_export: bool = Field(..., description="Whether the dataset is ready for export")
    ready_for_import: bool = Field(..., description="Whether the dataset is ready for import")
    size: int = Field(..., description="Dataset size in bytes")
    metadata: dict | None = Field(None, description="Dataset metadata")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "63f983fe-f2c7-4054-a0b1-6aab8a355a12",
                "format": "datumaro_v2",
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

    @model_validator(mode="before")
    @classmethod
    def populate_metadata(cls, data: object) -> object:
        if isinstance(data, StagedDataset):
            return {
                "id": data.id,
                "compressed": data.compressed,
                "size": data.size,
                "format": data.format,
                "ready_for_export": data.compressed,
                "ready_for_import": data.format == DatasetFormat.DATUMARO_V2 and not data.compressed,
                "metadata": data.metadata,
            }
        return data
