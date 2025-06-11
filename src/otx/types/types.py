"""Typing hints for OTX."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from otx.backend.native.models.base import OTXModel
from otx.data.entity.torch import OTXDataItem

METRICS = dict[str, float]
ANNOTATIONS = list[OTXDataItem]
MODEL = OTXModel  # TODO(ashwinvaidya17): Temporary till model is properly defined
DATA = Any  # TODO(ashwinvaidya17): Temporary till data is properly defined
