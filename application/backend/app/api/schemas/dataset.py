# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.core.models import BaseRequiredIDModel
from app.models import DatasetFormat, DatasetItemSubset, StagedDataset
from app.models.dataset import DatasetMetadata


class DatasetFilters(BaseModel):
    labels: list[str] | None = Field(
        None,
        description="List of labels to consider during import or export; any annotation with labels not present in "
        "the list will be filtered out; if the parameter is unspecified (null), then all labels will be considered",
    )
    subsets: list[DatasetItemSubset] | None = Field(
        None,
        description="List of subsets to consider during import or export; any item assigned a subset not present in "
        "the list will be filtered out; if the parameter is unspecified (null), then all subsets will be considered",
    )
    include_unannotated: bool = Field(True, description="Whether to include unannotated items from the dataset")

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
    metadata: DatasetMetadata | None = Field(None, description="Dataset metadata")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "63f983fe-f2c7-4054-a0b1-6aab8a355a12",
                "format": "geti",
                "compressed": False,
                "ready_for_export": False,
                "ready_for_import": True,
                "size": 987654321,
                "metadata": {
                    "num_images": 2000,
                    "num_frames": 800,
                    "num_videos": 10,
                    "annotation_type": "bounding_box",
                    "num_annotations": 10000,
                    "num_annotated_images": 1300,
                    "num_annotated_frames": 500,
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
                "ready_for_import": data.format == DatasetFormat.GETI and not data.compressed,
                "metadata": data.metadata,
            }
        return data


class MediaCountsView(BaseModel):
    """Media counts"""

    images: int = Field(0, description="Number of images in dataset")
    videos: int = Field(0, description="Number of videos in dataset")
    video_frames: int = Field(0, description="Number of video frames in dataset")

    model_config = {"json_schema_extra": {"example": {"images": 10, "videos": 3, "video_frames": 312}}}


class InstancesPerLabelView(BaseModel):
    label_id: UUID | None = Field(..., description="Unique identifier of label")
    instances: int = Field(0, description="Number of annotation instances with this label in dataset")

    model_config = {
        "json_schema_extra": {"example": {"label_id": "5fffd195-7766-4171-8efe-4064a6eb0e95", "instances": 24}}
    }


class AnnotationCountsView(BaseModel):
    annotated_images: int = Field(0, description="Number of annotated images in dataset")
    annotated_videos: int = Field(0, description="Number of annotated videos in dataset")
    annotated_video_frames: int = Field(0, description="Number of annotated video frames in dataset")
    instances: int = Field(0, description="Number of annotation shapes in dataset")
    instances_per_label: list[InstancesPerLabelView] = Field(
        [], description="List of number of annotation shapes per label"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "annotated_images": 8,
                "annotated_videos": 2,
                "annotated_video_frames": 29,
                "instances": 56,
                "instances_per_label": [
                    {"label_id": "5fffd195-7766-4171-8efe-4064a6eb0e95", "instances": 24},
                    {"label_id": "20f1defc-8d40-47ff-9f9d-8e82f0ef224d", "instances": 32},
                ],
            }
        }
    }


class DatasetStatisticsView(BaseModel):
    """
    Dataset statistics
    """

    media_counts: MediaCountsView = Field(..., description="Number of media per media type in dataset")
    annotations_counts: AnnotationCountsView = Field(
        ..., description="Number of annotated media per media type and labels"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "media_counts": {"images": 10, "videos": 3, "video_frames": 312},
                "annotations_counts": {
                    "annotated_images": 8,
                    "annotated_videos": 2,
                    "annotated_video_frames": 29,
                    "instances": 56,
                    "instances_per_label": [
                        {"label_id": "5fffd195-7766-4171-8efe-4064a6eb0e95", "instances": 24},
                        {"label_id": "20f1defc-8d40-47ff-9f9d-8e82f0ef224d", "instances": 32},
                    ],
                },
            }
        }
    }
