# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for model utility functions."""

from __future__ import annotations

import ssl
import urllib.request
from unittest.mock import MagicMock, patch

import certifi
import pytest

from otx.backend.native.models.utils.utils import load_from_http


def test_load_from_http_uses_certifi_ssl(mocker):
    """load_from_http should install an HTTPS opener using certifi's CA bundle."""
    installed_opener = None

    def capture_install_opener(opener):
        nonlocal installed_opener
        installed_opener = opener

    mocker.patch("urllib.request.install_opener", side_effect=capture_install_opener)
    mock_load_url = mocker.patch("otx.backend.native.models.utils.utils.load_url", return_value={"state_dict": {}})
    mocker.patch("otx.backend.native.models.utils.utils.get_dist_info", return_value=(0, 1))

    load_from_http("https://example.com/model.pth", map_location="cpu")

    assert installed_opener is not None, "install_opener was not called"
    mock_load_url.assert_called_once()
