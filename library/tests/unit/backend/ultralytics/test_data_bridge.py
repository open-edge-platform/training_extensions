# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Ultralytics data bridge."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch
from torchvision import tv_tensors

from getitune.backend.ultralytics.data import UltralyticsDatasetAdapter, ultralytics_collate_fn
from getitune.data.dataset.base import VisionDataset
from getitune.data.entity.base import ImageInfo


def test_dataset_adapter_keeps_float_images_and_converts_boxes() -> None:
    image = tv_tensors.Image(torch.rand(3, 16, 20, dtype=torch.float32))
    bboxes = tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
        torch.tensor([[2.0, 4.0, 10.0, 12.0]], dtype=torch.float32),
        format=tv_tensors.BoundingBoxFormat.XYXY,
        canvas_size=(16, 20),
    )
    sample = SimpleNamespace(
        image=image,
        bboxes=bboxes,
        label=torch.tensor([1]),
        img_info=ImageInfo(  # pyrefly: ignore[no-matching-overload]
            img_idx=0,
            img_shape=(12, 18),
            ori_shape=(16, 20),
            padding=(1, 2, 3, 4),
        ),
    )

    dataset = MagicMock(spec=VisionDataset)
    dataset.__len__.return_value = 1
    dataset.__getitem__.return_value = sample

    adapter = UltralyticsDatasetAdapter(dataset)
    result = adapter[0]

    assert result["img"] is image
    assert result["img"].dtype == torch.float32
    np.testing.assert_allclose(result["bboxes"], np.array([[0.3, 0.5, 0.4, 0.5]], dtype=np.float32))
    assert result["cls"].shape == (1, 1)
    assert result["ori_shape"] == (16, 20)
    assert result["resized_shape"] == (12, 18)
    assert result["ratio_pad"] == ((12 / 16, 18 / 20), (2, 1))


def test_ultralytics_collate_fn_matches_expected_detection_contract() -> None:
    batch = [
        {
            "img": torch.rand(3, 16, 16),
            "cls": np.array([[0.0], [1.0]], dtype=np.float32),
            "bboxes": np.array([[0.5, 0.5, 0.4, 0.4], [0.2, 0.2, 0.1, 0.1]], dtype=np.float32),
            "ori_shape": (16, 16),
            "resized_shape": (16, 16),
            "ratio_pad": ((1.0, 1.0), (0, 0)),
            "im_file": "a.jpg",
        },
        {
            "img": torch.rand(3, 16, 16),
            "cls": np.array([[2.0]], dtype=np.float32),
            "bboxes": np.array([[0.4, 0.4, 0.3, 0.3]], dtype=np.float32),
            "ori_shape": (16, 16),
            "resized_shape": (16, 16),
            "ratio_pad": ((1.0, 1.0), (0, 0)),
            "im_file": "b.jpg",
        },
    ]

    collated = ultralytics_collate_fn(batch)

    assert collated["img"].shape == (2, 3, 16, 16)
    assert collated["cls"].shape == (3, 1)
    assert collated["bboxes"].shape == (3, 4)
    assert torch.equal(collated["batch_idx"], torch.tensor([0.0, 0.0, 1.0]))
    assert collated["im_file"] == ["a.jpg", "b.jpg"]
