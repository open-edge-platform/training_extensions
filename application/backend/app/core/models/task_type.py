# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from enum import StrEnum


class TaskType(StrEnum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
