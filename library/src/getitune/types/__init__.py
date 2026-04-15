# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module reserved for definitions used in Geti Tune."""

import os
from pathlib import Path
from typing import Union

from typing_extensions import TypeAlias

from getitune.types.label import HLabelInfo, LabelInfo, NullLabelInfo, SegLabelInfo
from getitune.types.task import OTXTaskType

__all__ = [
    "HLabelInfo",
    # label_info
    "LabelInfo",
    "NullLabelInfo",
    # task_type
    "OTXTaskType",
    "SegLabelInfo",
]

PathLike: TypeAlias = Union[str, Path, os.PathLike]
