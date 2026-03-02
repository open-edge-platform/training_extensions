# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from uuid import UUID

from pydantic import Field, model_validator

from pydantic import BaseModel

from app.core.models import BaseRequiredIDModel
from app.models.base import BaseEntity


class AnnotationType(StrEnum):
    BOUNDING_BOX = "bounding_box"
    POLYGON = "polygon"
    LABEL = "label"
    UNKNOWN = "unknown"


class DatasetFormat(StrEnum):
    COCO = "coco"
    DATUMARO_V1 = "datumaro_v1"
    GETI = "geti"
    VOC = "voc"
    YOLO = "yolo"
    UNKNOWN = "unknown"


class DatasetMetadata(BaseModel):
    num_items: int
    annotation_type: AnnotationType
    num_annotations: int
    labels: list[str]


class StagedDataset(BaseRequiredIDModel):
    filename: str
    compressed: bool
    format: DatasetFormat
    size: int
    metadata: DatasetMetadata | None = None


class MediaCounts(BaseEntity):
    images: int = 0
    videos: int = 0
    video_frames: int = 0


class InstancesPerLabel(BaseEntity):
    label_id: UUID
    instances: int = 0

    @model_validator(mode="before")
    @classmethod
    def populate_instances_per_label(cls, data: object) -> object:
        if isinstance(data, dict):
            return {
                "label_id": UUID(str(data.get("label_id"))),
                "instances": data.get("instances", 0),
            }
        return data


class AnnotationCounts(BaseEntity):
    annotated_images: int = 0
    annotated_videos: int = 0
    annotated_video_frames: int = 0
    instances: int = 0
    instances_per_label: list[InstancesPerLabel] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def populate_annotation_counts(cls, data: object) -> object:
        if isinstance(data, dict):
            return {
                "annotated_images": data.get("annotated_images", 0),
                "annotated_videos": data.get("annotated_videos", 0),
                "annotated_video_frames": data.get("annotated_video_frames", 0),
                "instances": data.get("instances", 0),
                "instances_per_label": [
                    InstancesPerLabel.model_validate(item) for item in data.get("instances_per_label", [])
                ],
            }
        return data


class DatasetStatistics(BaseEntity):
    media_counts: MediaCounts = Field(default_factory=MediaCounts)
    annotations_counts: AnnotationCounts = Field(default_factory=AnnotationCounts)

    @model_validator(mode="before")
    @classmethod
    def populate_dataset_statistics(cls, data: object) -> object:
        if isinstance(data, dict):
            return {
                "media_counts": MediaCounts.model_validate(data),
                "annotations_counts": AnnotationCounts.model_validate(data),
            }
        return data
