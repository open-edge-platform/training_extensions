# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for sample entity classes."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
import torch
from datumaro import DatasetItem
from datumaro.components.annotation import Label
from datumaro.components.media import Image
from torchvision import tv_tensors

from otx.data.entity.base import ImageInfo
from otx.data.entity.sample import ClassificationSample, OTXSample


class TestOTXSample:
    """Test OTXSample base class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock sample for testing
        self.sample = OTXSample()

    def test_as_tv_image_with_tv_image(self):
        """Test as_tv_image when image is already tv_tensors.Image."""
        tv_image = tv_tensors.Image(torch.randn(3, 224, 224))
        self.sample.image = tv_image

        # Should not change anything
        self.sample.as_tv_image()
        assert isinstance(self.sample.image, tv_tensors.Image)
        assert torch.equal(self.sample.image, tv_image)

    def test_as_tv_image_with_numpy_array(self):
        """Test as_tv_image with numpy array."""
        np_image = np.random.rand(3, 224, 224).astype(np.float32)
        self.sample.image = np_image

        self.sample.as_tv_image()

        assert isinstance(self.sample.image, tv_tensors.Image)
        assert torch.allclose(self.sample.image, torch.from_numpy(np_image))

    def test_as_tv_image_with_torch_tensor(self):
        """Test as_tv_image with torch.Tensor."""
        tensor_image = torch.randn(3, 224, 224)
        self.sample.image = tensor_image

        self.sample.as_tv_image()

        assert isinstance(self.sample.image, tv_tensors.Image)
        assert torch.equal(self.sample.image, tensor_image)

    def test_as_tv_image_with_invalid_type(self):
        """Test as_tv_image with invalid image type raises ValueError."""
        self.sample.image = "invalid_image"

        with pytest.raises(ValueError, match="OTXSample must have an image"):
            self.sample.as_tv_image()

    def test_img_info_property_with_image(self):
        """Test img_info property creates ImageInfo from image."""
        self.sample.image = torch.randn(3, 224, 224)

        img_info = self.sample.img_info

        assert isinstance(img_info, ImageInfo)
        assert img_info.img_idx == 0
        assert img_info.img_shape == (3, 224)  # First two dimensions
        assert img_info.ori_shape == (3, 224)

    def test_img_info_setter(self):
        """Test setting img_info manually."""
        custom_info = ImageInfo(img_idx=5, img_shape=(100, 200), ori_shape=(100, 200))

        self.sample.img_info = custom_info

        assert self.sample.img_info is custom_info
        assert self.sample.img_info.img_idx == 5


class TestClassificationSample:
    """Test ClassificationSample class."""

    def test_inheritance(self):
        """Test that ClassificationSample inherits from OTXSample."""
        sample = ClassificationSample(
            image=np.random.rand(3, 224, 224).astype(np.uint8), label=torch.tensor(1)
        )

        assert isinstance(sample, OTXSample)

    def test_init_with_numpy_image_and_tensor_label(self):
        """Test initialization with numpy image and tensor label."""
        image = np.random.rand(3, 224, 224).astype(np.uint8)
        label = torch.tensor(1)

        sample = ClassificationSample(image=image, label=label)

        assert np.array_equal(sample.image, image)
        assert torch.equal(sample.label, label)

    def test_init_with_tv_image(self):
        """Test initialization with tv_tensors.Image."""
        image = tv_tensors.Image(torch.randn(3, 224, 224))
        label = torch.tensor(0)

        sample = ClassificationSample(image=image, label=label)

        assert torch.equal(sample.image, image)
        assert torch.equal(sample.label, label)

    def test_from_dm_item_with_image_and_annotation(self):
        """Test from_dm_item with image and annotation."""
        # Mock DatasetItem
        mock_item = Mock(spec=DatasetItem)

        # Mock image
        mock_media = Mock(spec=Image)
        mock_media.data = np.random.rand(224, 224, 3).astype(np.uint8)
        mock_item.media_as.return_value = mock_media

        # Mock annotation
        mock_annotation = Mock(spec=Label)
        mock_annotation.label = 2
        mock_item.annotations = [mock_annotation]

        sample = ClassificationSample.from_dm_item(mock_item)

        assert isinstance(sample, ClassificationSample)
        assert np.array_equal(sample.image, mock_media.data)
        assert torch.equal(sample.label, torch.tensor(2, dtype=torch.long))

        # Check img_info
        assert isinstance(sample._img_info, ImageInfo)
        assert sample._img_info.img_idx == 0
        assert sample._img_info.img_shape == (224, 224)
        assert sample._img_info.ori_shape == (224, 224)

    def test_from_dm_item_without_annotation(self):
        """Test from_dm_item without annotations."""
        # Mock DatasetItem without annotations
        mock_item = Mock(spec=DatasetItem)

        # Mock image
        mock_media = Mock(spec=Image)
        mock_media.data = np.random.rand(100, 100, 3).astype(np.uint8)
        mock_item.media_as.return_value = mock_media

        # No annotations
        mock_item.annotations = []

        sample = ClassificationSample.from_dm_item(mock_item)

        assert isinstance(sample, ClassificationSample)
        assert np.array_equal(sample.image, mock_media.data)
        # When no annotation, from_dm_item should return tensor(-1) as default
        assert torch.equal(sample.label, torch.tensor(-1, dtype=torch.long))

    def test_label_property_override(self):
        """Test that ClassificationSample has actual label property (not None)."""
        sample = ClassificationSample(
            image=np.random.rand(3, 224, 224).astype(np.uint8), label=torch.tensor(42)
        )

        assert sample.label is not None
        assert torch.equal(sample.label, torch.tensor(42))
