# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module reserved for definitions used in getitune."""

import os
from pathlib import Path
from typing import Union

from typing_extensions import TypeAlias

from getitune.types.label import HLabelInfo, LabelInfo, NullLabelInfo, SegLabelInfo
from getitune.types.task import TaskType

__all__ = [
    "HLabelInfo",
    "LabelInfo",
    "NullLabelInfo",
    "SegLabelInfo",
    "TaskType",
]

PathLike: TypeAlias = Union[str, Path, os.PathLike]
