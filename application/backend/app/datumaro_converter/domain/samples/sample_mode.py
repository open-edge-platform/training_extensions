# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, auto


class SampleMode(Enum):
    """Defines the mode of sample creation, which influence what types of samples are generated from dataset items."""

    TRAINING = auto()
    IMPORT_EXPORT = auto()
