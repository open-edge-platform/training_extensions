# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum

from pydantic import BaseModel

from app.core.models import BaseRequiredIDModel


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
