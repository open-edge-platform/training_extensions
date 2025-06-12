"""Typing hints for OTX."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from otx.backend.native.models.base import OTXModel
from otx.data.entity import OTXDataItem
from otx.data.module import OTXDataModule

METRICS = dict[str, float]
ANNOTATIONS = list[OTXDataItem]
MODEL = OTXModel  # TODO(ashwinvaidya17, kprokofi): Temporary till model is properly defined
DATA = OTXDataModule  # TODO(ashwinvaidya17, kprokofi): Temporary till data is properly defined
