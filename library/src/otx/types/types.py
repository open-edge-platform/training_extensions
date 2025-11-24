"""Typing hints for OTX."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from otx.backend.native.models.base import OTXModel
from otx.backend.openvino.models.base import OVModel
from otx.data.entity import OTXDataItem
from otx.data.module import OTXDataModule
from otx.types import PathLike

METRICS = dict[str, float]
ANNOTATIONS = list[OTXDataItem]
MODEL = OTXModel | OVModel | PathLike
DATA = OTXDataModule | PathLike
