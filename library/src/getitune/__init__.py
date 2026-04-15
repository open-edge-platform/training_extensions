# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Geti Tune - Train, Evaluate, Optimize, Deploy Computer Vision Models."""

__version__ = "0.1.0"

import os
from pathlib import Path

from getitune.types import *  # noqa: F403

# Set the value of HF_HUB_CACHE to set the cache folder that stores the pretrained weights for timm and huggingface.
# Refer: huggingface_hub/constants.py::HF_HUB_CACHE
# Default, Pretrained weight is saved into ~/.cache/torch/hub/checkpoints
os.environ["HF_HUB_CACHE"] = os.getenv(
    "HF_HUB_CACHE",
    str(Path.home() / ".cache" / "torch" / "hub" / "checkpoints"),
)
# Set the value of ONEDNN_PRIMITIVE_CACHE_CAPACITY to set the cache capacity for oneDNN primitives.
# It will be ignored if no XPU devices are available.
os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = "10000"

LIBRARY_LOGO: str = """
 ██████╗ ███████╗████████╗██╗    ████████╗██╗   ██╗███╗   ██╗███████╗
██╔════╝ ██╔════╝╚══██╔══╝██║    ╚══██╔══╝██║   ██║████╗  ██║██╔════╝
██║  ███╗█████╗     ██║   ██║       ██║   ██║   ██║██╔██╗ ██║█████╗
██║   ██║██╔══╝     ██║   ██║       ██║   ██║   ██║██║╚██╗██║██╔══╝
╚██████╔╝███████╗   ██║   ██║       ██║   ╚██████╔╝██║ ╚████║███████╗
 ╚═════╝ ╚══════╝   ╚═╝   ╚═╝       ╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚══════╝
"""
