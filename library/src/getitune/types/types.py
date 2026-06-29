"""Typing hints for getitune."""

# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from getitune.backend.lightning.models.base import LightningModel
from getitune.backend.openvino.models.base import OVModel
from getitune.data.entity import BaseSample
from getitune.data.module import DataModule
from getitune.types import PathLike

try:
    from getitune.backend.ultralytics.models.base import UltralyticsModel
except ImportError:  # ultralytics not installed
    UltralyticsModel = None  # type: ignore[assignment, misc]

METRICS = dict[str, float]
ANNOTATIONS = list[BaseSample]
if UltralyticsModel is not None:
    MODEL = LightningModel | OVModel | UltralyticsModel | PathLike
else:
    MODEL = LightningModel | OVModel | PathLike
DATA = DataModule | PathLike
