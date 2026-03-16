# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Datumaro converter for loading high-bit-depth images (uint16, int16, float32).

The built-in ``ImagePathToImageConverter`` always decodes to ``UInt8``.
This module registers a ``HighBitImagePathConverter`` that preserves the
native bit depth so that ``with_image_dtype(SampleCls, "uint16")`` triggers
correct 16-bit loading via Datumaro's converter registry.
"""

from typing import Any

import cv2
import numpy as np
import polars as pl
from datumaro.experimental.converters.base import Converter
from datumaro.experimental.converters.registry import AttributeSpec, converter
from datumaro.experimental.fields import ImageField
from datumaro.experimental.fields.images import ImagePathField

_HIGHBIT_DTYPES = {pl.UInt16(), pl.Int16(), pl.Float32()}


@converter(lazy=True)
class HighBitImagePathConverter(Converter):
    """Load images from paths preserving native bit depth (uint16, int16, float32)."""

    input_path: AttributeSpec[ImagePathField]
    output_image: AttributeSpec[ImageField]

    def filter_output_spec(self) -> bool:
        """Accept only non-uint8 target dtypes; let the default converter handle uint8."""
        target_dtype = self.output_image.field.dtype
        if target_dtype not in _HIGHBIT_DTYPES:
            return False
        self.output_image = AttributeSpec(
            name=self.output_image.name,
            field=ImageField(
                semantic=self.input_path.field.semantic,
                dtype=target_dtype,
                format=self.output_image.field.format or "RGB",
                channels_first=self.output_image.field.channels_first,
            ),
        )
        return True

    def convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """Load images from disk with ``cv2.IMREAD_UNCHANGED`` to preserve bit depth."""
        input_col = self.input_path.name
        output_col = self.output_image.name
        output_format = self.output_image.field.format

        image_data: list[Any] = []
        image_shapes: list[list[int]] = []
        for path in df[input_col]:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                msg = f"Failed to load image: {path}"
                raise FileNotFoundError(msg)
            # Ensure 3-channel for RGB/BGR
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            if output_format == "RGB" and img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_data.append(img.flatten())
            image_shapes.append(list(img.shape))

        image_schema = self.output_image.field.to_polars_schema("image")
        return df.clone().with_columns(
            [
                pl.Series(output_col, image_data, dtype=image_schema["image"]),
                pl.Series(output_col + "_shape", image_shapes, dtype=image_schema["image_shape"]),
            ]
        )
