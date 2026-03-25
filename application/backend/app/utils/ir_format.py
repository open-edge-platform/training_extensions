# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""OpenVINO IR adapter for float32 [0-1] input models."""

from __future__ import annotations

from typing import Any

from model_api.adapters import OpenvinoAdapter


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
