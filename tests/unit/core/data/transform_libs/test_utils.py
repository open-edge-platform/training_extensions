# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from otx.core.data.transform_libs.utils import get_image_shape, rescale_keypoints, rescale_size, to_np_image


@pytest.mark.parametrize(("img", "expected_shape"), [(np.zeros((1, 2, 3)), (1, 2)), (torch.zeros((1, 2, 3)), (2, 3))])
@pytest.mark.parametrize("is_list", [True, False])
def test_get_image_shape(img: np.ndarray | Tensor | list, is_list: bool, expected_shape: tuple[int, int]) -> None:
    if is_list:
        img = [img, img]

    results = get_image_shape(img)

    assert results == expected_shape


@pytest.mark.parametrize("img", [np.zeros((1, 2, 3)), torch.zeros((1, 2, 3))])
@pytest.mark.parametrize("is_list", [True, False])
def test_to_np_image(img: np.ndarray | Tensor | list, is_list: bool) -> None:
    results = to_np_image(img)

    if is_list:
        assert all(isinstance(r, np.ndarray) for r in results)
    else:
        assert isinstance(results, np.ndarray)


@pytest.mark.parametrize(
    ("size", "scale", "expected_size"),
    [
        ((100, 200), 0.5, (50, 100)),
        ((200, 100), 2, (400, 200)),
        ((200, 100), (300, 300), (300, 150)),
        ((200, 100), (50, 50), (50, 25)),
    ],
)
def test_rescale_size(size: tuple[int, int], scale: float, expected_size: tuple[int, int]) -> None:
    results = rescale_size(size, scale)

    assert results == expected_size


def test_rescale_keypoints():
    keypoints = np.array([[10, 20], [30, 40], [50, 60]], dtype=float)

    # Test with a single float scale factor
    scale_factor = 2.0
    rescaled_keypoints = rescale_keypoints(keypoints.copy(), scale_factor)
    expected_keypoints = np.array([[20, 40], [60, 80], [100, 120]])
    np.testing.assert_array_almost_equal(rescaled_keypoints, expected_keypoints)

    # Test with a tuple scale factor
    scale_factor = (2.0, 0.5)
    rescaled_keypoints = rescale_keypoints(keypoints.copy(), scale_factor)
    expected_keypoints = np.array([[5, 40], [15, 80], [25, 120]])
    np.testing.assert_array_almost_equal(rescaled_keypoints, expected_keypoints)

    # Test with a different tuple scale factor
    scale_factor = (0.5, 2.0)
    rescaled_keypoints = rescale_keypoints(keypoints.copy(), scale_factor)
    expected_keypoints = np.array([[20, 10], [60, 20], [100, 30]])
    np.testing.assert_array_almost_equal(rescaled_keypoints, expected_keypoints)

    # Test with a single float scale factor of 1.0 (no scaling)
    scale_factor = 1.0
    rescaled_keypoints = rescale_keypoints(keypoints.copy(), scale_factor)
    expected_keypoints = keypoints
    np.testing.assert_array_almost_equal(rescaled_keypoints, expected_keypoints)
