# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for utils for OTX data module."""

from __future__ import annotations

from collections import defaultdict
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from datumaro.components.annotation import Bbox, Label
from datumaro.components.dataset import Dataset as DmDataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image

from otx.data.utils import utils as target_file
from otx.data.utils.utils import (
    adapt_input_size_to_dataset,
    compute_robust_dataset_statistics,
    compute_robust_scale_statistics,
    compute_robust_statistics,
    get_adaptive_num_workers,
    get_idx_list_per_classes,
    import_object_from_module,
)


def test_compute_robust_statistics():
    values = np.array([])
    stat = compute_robust_statistics(values)
    assert len(stat) == 0

    values = np.array([0.5, 1, 1.5])
    stat = compute_robust_statistics(values)
    assert np.isclose(stat["avg"], 1.0)
    assert np.isclose(stat["min"], 0.5)
    assert np.isclose(stat["max"], 1.5)

    values = np.random.rand(10)
    stat = compute_robust_statistics(values)
    assert np.isclose(stat["min"], np.min(values))
    assert np.isclose(stat["max"], np.max(values))
    assert stat["min"] <= stat["robust_min"]
    assert stat["max"] <= stat["robust_max"]


def test_compute_robust_scale_statistics():
    scales = np.array([])
    stat = compute_robust_scale_statistics(scales)
    assert len(stat) == 0

    scales = np.array([0.5, 1, 2])
    stat = compute_robust_scale_statistics(scales)
    assert np.isclose(stat["avg"], 1.0)
    assert np.isclose(stat["min"], 0.5)
    assert np.isclose(stat["max"], 2.0)

    scales = np.random.rand(10)
    stat = compute_robust_scale_statistics(scales)
    assert np.isclose(stat["min"], np.min(scales))
    assert np.isclose(stat["max"], np.max(scales))
    assert stat["min"] <= stat["robust_min"]
    assert stat["max"] <= stat["robust_max"]


def make_media(shape: tuple[int, int, int]):
    np_img = np.zeros(shape=shape, dtype=np.uint8)
    np_img[:, :, 0] = 0  # Set 0 for B channel
    np_img[:, :, 1] = 1  # Set 1 for G channel
    np_img[:, :, 2] = 2  # Set 2 for R channel

    _, np_bytes = cv2.imencode(".png", np_img)
    media = Image.from_bytes(np_bytes.tobytes())
    media.path = ""

    return media


@pytest.fixture()
def mock_dataset() -> DmDataset:
    return DmDataset.from_iterable(
        [
            DatasetItem(
                id="1",
                subset="train",
                media=make_media((50, 50, 3)),
                annotations=[
                    Bbox(x=0, y=0, w=5, h=5, label=0),
                ],
            ),
            DatasetItem(
                id="2",
                subset="train",
                media=make_media((100, 100, 3)),
                annotations=[
                    Bbox(x=0, y=0, w=10, h=10, label=0),
                    Bbox(x=10, y=10, w=20, h=20, label=0),
                ],
            ),
            DatasetItem(
                id="3",
                subset="train",
                media=make_media((200, 200, 3)),
                annotations=[],
            ),
        ],
    )


def test_compute_robuste_dataset_statistics(mock_dataset):
    subset = mock_dataset.get_subset("train")

    stat = compute_robust_dataset_statistics(subset, max_samples=0)
    assert stat["image"] == {}
    assert stat["annotation"] == {}
    stat = compute_robust_dataset_statistics(subset, max_samples=-1)
    assert stat["image"] == {}
    assert stat["annotation"] == {}

    stat = compute_robust_dataset_statistics(subset)
    assert np.isclose(stat["image"]["height"]["avg"], 100)
    assert np.isclose(stat["image"]["width"]["avg"], 100)
    assert np.isclose(stat["annotation"]["num_per_image"]["avg"], 1.0)
    assert np.isclose(stat["annotation"]["size_of_shape"]["avg"], 10.0)


def test_adapt_input_size_to_dataset(mocker):
    mock_stat = mocker.patch.object(target_file, "compute_robust_dataset_statistics")

    mock_stat.return_value = {"image": {}, "annotation": {}}
    mock_dataset = MagicMock()
    mock_dataset.subsets.return_value = {}
    with pytest.raises(ValueError, match="Dataset does not have 'train' subset. Cannot compute dataset statistics."):
        input_size = adapt_input_size_to_dataset(dataset=mock_dataset)

    mock_stat.return_value = {"image": {}, "annotation": {}}
    input_size = adapt_input_size_to_dataset(dataset=MagicMock())
    assert input_size == (0, 0)

    mock_stat.return_value = {
        "image": {
            "height": {"robust_max": 150},
            "width": {"robust_max": 200},
        },
        "annotation": {},
    }
    input_size = adapt_input_size_to_dataset(dataset=MagicMock())
    assert input_size == (150, 200)

    mock_stat.return_value = {
        "image": {
            "height": {"robust_max": 150},
            "width": {"robust_max": 200},
        },
        "annotation": {},
    }
    input_size = adapt_input_size_to_dataset(dataset=MagicMock(), input_size_multiplier=32)
    assert input_size == (160, 224)

    mock_stat.return_value = {
        "image": {
            "height": {"robust_max": 224},
            "width": {"robust_max": 240},
        },
        "annotation": {"size_of_shape": {"robust_min": 64}},
    }
    input_size = adapt_input_size_to_dataset(dataset=MagicMock())
    assert input_size == (256, 274)

    mock_stat.return_value = {
        "image": {
            "height": {"robust_max": 1024},
            "width": {"robust_max": 1200},
        },
        "annotation": {"size_of_shape": {"robust_min": 64}},
    }
    input_size = adapt_input_size_to_dataset(dataset=MagicMock())
    assert input_size == (512, 600)

    mock_stat.return_value = {
        "image": {
            "height": {"robust_max": 2045},
            "width": {"robust_max": 2045},
        },
        "annotation": {"size_of_shape": {"robust_min": 64}},
    }
    input_size = adapt_input_size_to_dataset(dataset=MagicMock())
    assert input_size == (1022, 1022)


@pytest.mark.parametrize("num_dataloader", [1, 2, 4])
def test_get_adaptive_num_workers(mocker, num_dataloader):
    num_gpu = 5
    mock_torch = mocker.patch.object(target_file, "torch")
    mock_torch.cuda.device_count.return_value = num_gpu

    num_cpu = 20
    mocker.patch.object(target_file, "cpu_count", return_value=num_cpu)

    assert get_adaptive_num_workers(num_dataloader) == num_cpu // (num_gpu * num_dataloader)


def test_get_adaptive_num_workers_no_gpu(mocker):
    num_gpu = 0
    mock_torch = mocker.patch.object(target_file, "torch")
    mock_torch.cuda.device_count.return_value = num_gpu

    num_cpu = 20
    mocker.patch.object(target_file, "cpu_count", return_value=num_cpu)

    assert get_adaptive_num_workers() is None


@pytest.fixture()
def fxt_dm_dataset() -> DmDataset:
    dataset_items = [
        DatasetItem(
            id=f"item00{i}_0",
            subset="train",
            media=None,
            annotations=[
                Label(label=0),
            ],
        )
        for i in range(1, 101)
    ] + [
        DatasetItem(
            id=f"item00{i}_1",
            subset="train",
            media=None,
            annotations=[
                Label(label=1),
            ],
        )
        for i in range(1, 9)
    ]

    return DmDataset.from_iterable(dataset_items, categories=["0", "1"])


def test_get_idx_list_per_classes(fxt_dm_dataset):
    # Call the function under test
    result = get_idx_list_per_classes(fxt_dm_dataset)

    # Assert the expected output
    expected_result = defaultdict(list)
    expected_result[0] = list(range(100))
    expected_result[1] = list(range(100, 108))
    assert result == expected_result

    # Call the function under test with use_string_label
    result = get_idx_list_per_classes(fxt_dm_dataset, use_string_label=True)

    # Assert the expected output
    expected_result = defaultdict(list)
    expected_result["0"] = list(range(100))
    expected_result["1"] = list(range(100, 108))
    assert result == expected_result


def test_import_object_from_module():
    obj_path = "otx.data.utils.get_idx_list_per_classes"
    obj = import_object_from_module(obj_path)
    assert obj == get_idx_list_per_classes
