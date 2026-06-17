# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""getitune - Train, Evaluate, Optimize, Deploy Computer Vision Models."""

from pathlib import Path

try:
    from importlib.metadata import version

    __version__ = version("getitune")
except Exception:
    # Fallback: read VERSION file when working from source (not installed)
    version_file = Path(__file__).parent.parent.parent / "VERSION"
    __version__ = version_file.read_text().strip() if version_file.exists() else "0.0.0"

import os

from getitune.types import *  # noqa: F403

# The 'PRETRAINED_WEIGHTS_CACHE_DIR' env variable controls where to cache pretrained weights for the majority of
# the models. The default location is ~/.cache/torch/hub/checkpoints
os.environ["PRETRAINED_WEIGHTS_CACHE_DIR"] = os.getenv(
    "PRETRAINED_WEIGHTS_CACHE_DIR",
    str(Path.home() / ".cache" / "torch" / "hub" / "checkpoints"),
)

# The 'HF_HUB_CACHE' env variable controls where to cache pretrained weights for timm and huggingface models.
# If not explicitly set, it defaults to the same location of PRETRAINED_WEIGHTS_CACHE_DIR.
# Refer: huggingface_hub/constants.py::HF_HUB_CACHE
os.environ["HF_HUB_CACHE"] = os.getenv(
    "HF_HUB_CACHE",
    os.environ["PRETRAINED_WEIGHTS_CACHE_DIR"],
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
