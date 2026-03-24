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
    """OpenvinoAdapter that forces float32 input tensors (0-1 scale IRs)."""

    def embed_preprocessing(self, *args: Any, **kwargs: Any) -> None:
        kwargs["dtype"] = float
        _patch_pad_constant_type(super().embed_preprocessing, *args, **kwargs)


def _patch_pad_constant_type(embed_fn: Any, *args: Any, **kwargs: Any) -> None:
    """Call ``embed_fn`` while monkey-patching ``opset.pad`` to fix pad-value dtype.

    ModelAPI hardcodes pad constants as ``uint8``.  With f32 input this causes
    a type-mismatch error, so we temporarily wrap ``opset.pad`` to insert a
    ``Convert`` when the element types differ.
    """
    import model_api.adapters.utils as _mapi_utils

    _opset = _mapi_utils.opset
    _orig_pad = _opset.pad

    def _pad_with_type_cast(
        arg: Any,
        pads_begin: Any,
        pads_end: Any,
        pad_mode: str,
        arg_pad_value: Any = None,
        name: Any = None,
    ) -> Any:
        if arg_pad_value is not None:
            data_et = arg.get_element_type()
            pad_et = arg_pad_value.get_element_type()
            if data_et != pad_et:
                arg_pad_value = _opset.convert(arg_pad_value, data_et)
        return _orig_pad(arg, pads_begin, pads_end, pad_mode, arg_pad_value, name)

    _opset.pad = _pad_with_type_cast  # pyrefly: ignore[bad-assignment]
    try:
        embed_fn(*args, **kwargs)
    finally:
        _opset.pad = _orig_pad


def needs_float32_input(model_xml_path: str | Path) -> bool:
    """Check whether the IR uses 0-1 normalisation scale (new OTX format).

    Args:
        model_xml_path: Path to the ``.xml`` IR file.

    Returns:
        True if all mean/scale values in rt_info are <= 1.0, False otherwise.
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
