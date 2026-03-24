# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Utilities for inspecting OpenVINO IR model metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import defusedxml.ElementTree as ET  # noqa: N817
from loguru import logger
from model_api.adapters import OpenvinoAdapter

# Values above this threshold mean the IR stores mean/std in uint8 (0-255) scale.
_UINT8_SCALE_THRESHOLD = 1.0


class FP32OpenvinoAdapter(OpenvinoAdapter):
    """OpenvinoAdapter that forces float32 input tensors.

    Used when the IR embeds mean/std in the 0-1 scale (new OTX format).
    Overrides ``embed_preprocessing`` so ModelAPI sets the input tensor to f32.
    """

    def embed_preprocessing(self, *args: Any, **kwargs: Any) -> None:
        """Force dtype to float so ModelAPI creates an f32 input tensor."""
        kwargs["dtype"] = float
        super().embed_preprocessing(*args, **kwargs)


def needs_float32_input(model_xml_path: str | Path) -> bool:
    """Return True if the IR expects float32 [0, 1] inputs (new OTX format).

    Reads ``mean_values`` and ``scale_values`` from the IR's ``<rt_info>``
    block.  If all values are <= 1.0 the model was exported with the new
    0-1 normalisation scale and callers must:

    * Pass images pre-scaled to ``float32 / 255``.
        * Load the model via ``FP32OpenvinoAdapter`` so that ModelAPI sets the
      input tensor type to ``f32``.

    If any value exceeds the threshold the IR uses the old uint8 scale
    (e.g. ``mean_values = 123.675 116.28 103.53``), and raw ``uint8``
    images should be passed as usual with the default ``OpenvinoAdapter``.

    When no ``mean_values`` / ``scale_values`` are present in the IR, the
    model has no embedded normalisation and raw ``uint8`` images are safe
    (returns ``False``).

    Args:
        model_xml_path: Path to the ``.xml`` file of the OpenVINO IR model.

    Returns:
        ``True``  → new 0-1 scale → use ``FP32OpenvinoAdapter`` + scale to float32.
        ``False`` → old uint8 scale (or no normalisation) → use default adapter + uint8 input.
    """
    try:
        tree = ET.parse(str(model_xml_path))
    except (OSError, ET.ParseError):
        logger.warning("Failed to parse IR XML '{}'; assuming uint8 input format", model_xml_path)
        return False
    root = tree.getroot()
    if root is None:
        return False
    rt_info = root.find("rt_info")
    if rt_info is None:
        return False

    def _parse_values(node: ET.Element, tag: str) -> list[float]:
        el = node.find(f".//{tag}")
        if el is None:
            return []
        raw = el.attrib.get("value", "").strip()
        if not raw:
            return []
        try:
            return [float(v) for v in raw.split()]
        except ValueError:
            return []

    mean_values = _parse_values(rt_info, "mean_values")
    scale_values = _parse_values(rt_info, "scale_values")

    all_values = mean_values + scale_values
    if not all_values:
        # No normalisation embedded — treat as uint8.
        return False

    return all(v <= _UINT8_SCALE_THRESHOLD for v in all_values)
