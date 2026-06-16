"""Typing hints for getitune."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from getitune.backend.lightning.models.base import LightningModel
from getitune.backend.openvino.models.base import OVModel
from getitune.data.entity import BaseSample
from getitune.data.module import DataModule
from getitune.types import PathLike

METRICS = dict[str, float]
ANNOTATIONS = list[BaseSample]
MODEL = LightningModel | OVModel | PathLike
DATA = DataModule | PathLike
