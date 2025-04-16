# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module reserved for definitions used in OTX."""

import os
from pathlib import Path
from typing import Union

from typing_extensions import TypeAlias

from .collate import CollateMode
from .device import DeviceType
from .image import ImageColorChannel
from .label import HLabelInfo, LabelInfo, NullLabelInfo, SegLabelInfo
from .task import OTXTaskType
from .transformer_libs import TransformLibType

__all__ = [
    "DeviceType",
    "ImageColorChannel",
    "LabelInfo",
    "HLabelInfo",
    "SegLabelInfo",
    "NullLabelInfo",
    "OTXTaskType",
    "CollateMode",
    "TransformLibType",
]

PathLike: TypeAlias = Union[str, Path, os.PathLike]
