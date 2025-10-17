# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import random


def random_color() -> str:
    """
    Generate random color.
    """
    red, green, blue = (
        random.randint(0, 255),  # noqa: S311 # nosec: B311
        random.randint(0, 255),  # noqa: S311 # nosec: B311
        random.randint(0, 255),  # noqa: S311 # nosec: B311
    )
    return f"#{red:02x}{green:02x}{blue:02x}"
