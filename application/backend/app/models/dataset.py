# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum

from app.core.models import BaseRequiredIDModel


class DatasetFormat(StrEnum):
    COCO = "coco"
    DATUMARO = "datumaro"
    VOC = "voc"
    YOLO = "yolo"
    UNKNOWN = "unknown"


class StagedDataset(BaseRequiredIDModel):
    compressed: bool
    format: DatasetFormat
    size: int
    metadata: dict | None = None
