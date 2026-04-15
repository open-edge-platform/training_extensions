"""Typing hints for Geti Tune."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from getitune.backend.native.models.base import OTXModel
from getitune.backend.openvino.models.base import OVModel
from getitune.data.entity import OTXSample
from getitune.data.module import OTXDataModule
from getitune.types import PathLike

METRICS = dict[str, float]
ANNOTATIONS = list[OTXSample]
MODEL = OTXModel | OVModel | PathLike
DATA = OTXDataModule | PathLike
