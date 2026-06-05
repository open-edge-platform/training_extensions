# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Smoke test: the package imports and exposes a version."""

import getitrack


def test_package_imports() -> None:
    assert hasattr(getitrack, "__version__")
    assert isinstance(getitrack.__version__, str)
