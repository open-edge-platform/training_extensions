# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum


class DatasetItemFormat(StrEnum):
    JPG = "jpg"
    PNG = "png"


class DatasetItemSubset(StrEnum):
    UNASSIGNED = "unassigned"
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
