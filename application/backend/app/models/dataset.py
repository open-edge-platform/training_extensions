# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum

from app.core.models import BaseRequiredIDModel


class DatasetFormat(StrEnum):
    COCO = "coco"
    DATUMARO_V1 = "datumaro_v1"
    DATUMARO_V2 = "datumaro_v2"
    VOC = "voc"
    YOLO = "yolo"
    UNKNOWN = "unknown"


class StagedDataset(BaseRequiredIDModel):
    filename: str
    compressed: bool
    format: DatasetFormat
    size: int
    metadata: dict | None = None
