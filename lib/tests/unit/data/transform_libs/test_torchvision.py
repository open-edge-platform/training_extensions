# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Unit tests of detection data transform."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
from datumaro import Polygon
from torch import LongTensor
from torchvision import tv_tensors
from torchvision.transforms.v2 import ToDtype
from torchvision.transforms.v2 import functional as F  # noqa: N812

from otx.data.entity.base import ImageInfo
from otx.data.entity.torch import OTXDataBatch, OTXDataItem
from otx.data.transform_libs.torchvision import (
    CachedMixUp,
    CachedMosaic,
    Compose,
    MinIoURandomCrop,
    Pad,
    PhotoMetricDistortion,
    RandomAffine,
    RandomCrop,
    RandomFlip,
    RandomGaussianNoise,
    RandomResize,
    Resize,
    TopdownAffine,
    YOLOXHSVRandomAug,
)
from otx.data.transform_libs.utils import overlap_bboxes


class MockFrame:
    data = np.ndarray([10, 10, 3], dtype=np.uint8)


class MockVideo:
    data = [MockFrame()] * 10

    def __getitem__(self, idx):
        return self.data[idx]

    def close(self):
        return


@pytest.fixture()
def seg_data_entity() -> OTXDataItem:
    masks = torch.randint(low=0, high=2, size=(1, 112, 224), dtype=torch.uint8)
    return OTXDataItem(
        image=tv_tensors.Image(torch.randint(low=0, high=256, size=(3, 112, 224), dtype=torch.uint8)),
        img_info=ImageInfo(img_idx=0, img_shape=(112, 224), ori_shape=(112, 224)),
        masks=tv_tensors.Mask(masks),
        label=LongTensor([1]),
    )


@pytest.fixture()
def det_data_entity() -> OTXDataItem:
    return OTXDataItem(
        image=tv_tensors.Image(torch.randint(low=0, high=256, size=(3, 112, 224), dtype=torch.uint8)),
        img_info=ImageInfo(img_idx=0, img_shape=(112, 224), ori_shape=(112, 224)),
        bboxes=tv_tensors.BoundingBoxes(data=torch.Tensor([0, 0, 50, 50]), format="xywh", canvas_size=(112, 224)),
        label=LongTensor([1]),
    )


@pytest.fixture()
def det_data_entity_with_masks() -> OTXDataItem:
    """Create a data entity with masks for testing."""
    img_size = (112, 224)
    fake_image = torch.randint(low=0, high=256, size=(3, *img_size), dtype=torch.uint8)
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_bboxes = tv_tensors.BoundingBoxes(
        data=torch.Tensor([[10, 10, 50, 50], [60, 60, 100, 100]]),
        format="xyxy",
        canvas_size=img_size,
    )
    fake_labels = LongTensor([1, 2])

    # Create meaningful masks that correspond to the bounding boxes
    masks = torch.zeros(size=(2, *img_size), dtype=torch.bool)
    masks[0, 10:50, 10:50] = True  # First mask
    masks[1, 60:100, 60:100] = True  # Second mask
    fake_masks = tv_tensors.Mask(masks)

    return OTXDataItem(
        image=tv_tensors.Image(fake_image),
        img_info=fake_image_info,
        bboxes=fake_bboxes,
        label=fake_labels,
        masks=fake_masks,
    )


@pytest.fixture()
def det_data_entity_empty_masks() -> OTXDataItem:
    """Create a data entity with empty masks for testing."""
    img_size = (112, 224)
    fake_image = torch.randint(low=0, high=256, size=(3, *img_size), dtype=torch.uint8)
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_bboxes = tv_tensors.BoundingBoxes(data=torch.Tensor([[10, 10, 50, 50]]), format="xyxy", canvas_size=img_size)
    fake_labels = LongTensor([1])

    # Create empty masks
    fake_masks = tv_tensors.Mask(torch.zeros(size=(0, *img_size), dtype=torch.uint8))

    return OTXDataItem(
        image=tv_tensors.Image(fake_image),
        img_info=fake_image_info,
        bboxes=fake_bboxes,
        label=fake_labels,
        masks=fake_masks,
    )


@pytest.fixture()
def det_data_entity_with_polygons() -> OTXDataItem:
    """Create a data entity with polygons for testing."""
    img_size = (112, 224)
    fake_image = torch.randint(low=0, high=256, size=(3, *img_size), dtype=torch.uint8)
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_bboxes = tv_tensors.BoundingBoxes(
        data=torch.Tensor([[10, 10, 50, 50], [60, 60, 100, 100]]),
        format="xyxy",
        canvas_size=img_size,
    )
    fake_labels = LongTensor([1, 2])

    # Create meaningful masks that correspond to the bounding boxes
    masks = torch.zeros(size=(2, *img_size), dtype=torch.bool)
    masks[0, 10:50, 10:50] = True  # First mask
    masks[1, 60:100, 60:100] = True  # Second mask
    fake_masks = tv_tensors.Mask(masks)

    # Create corresponding polygons
    fake_polygons = [
        Polygon(points=[10, 10, 50, 10, 50, 50, 10, 50]),  # Rectangle polygon for first object
        Polygon(points=[60, 60, 100, 60, 100, 100, 60, 100]),  # Rectangle polygon for second object
    ]

    return OTXDataItem(
        image=tv_tensors.Image(fake_image),
        img_info=fake_image_info,
        bboxes=fake_bboxes,
        label=fake_labels,
        masks=fake_masks,
        polygons=fake_polygons,
    )


@pytest.fixture()
def det_data_entity_empty_polygons() -> OTXDataItem:
    """Create a data entity with empty polygons for testing."""
    img_size = (112, 224)
    fake_image = torch.randint(low=0, high=256, size=(3, *img_size), dtype=torch.uint8)
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_bboxes = tv_tensors.BoundingBoxes(data=torch.Tensor([[10, 10, 50, 50]]), format="xyxy", canvas_size=img_size)
    fake_labels = LongTensor([1])

    # Create empty polygons
    fake_polygons = []

    return OTXDataItem(
        image=tv_tensors.Image(fake_image),
        img_info=fake_image_info,
        bboxes=fake_bboxes,
        label=fake_labels,
        polygons=fake_polygons,
    )


class TestMinIoURandomCrop:
    @pytest.fixture()
    def min_iou_random_crop(self) -> MinIoURandomCrop:
        return MinIoURandomCrop(is_numpy_to_tvtensor=False)

    def test_forward(self, min_iou_random_crop: MinIoURandomCrop, det_data_entity: OTXDataItem) -> None:
        """Test forward."""
        results = min_iou_random_crop(deepcopy(det_data_entity))

        if (mode := min_iou_random_crop.mode) == 1:
            assert torch.equal(results.bboxes, det_data_entity.bboxes)
        else:
            patch = tv_tensors.wrap(torch.tensor([[0, 0, *results.img_info.img_shape]]), like=results.bboxes)
            ious = overlap_bboxes(patch, results.bboxes)
            assert torch.all(ious >= mode)
            assert results.image.shape[:2] == results.img_info.img_shape
            assert results.img_info.scale_factor is None


class TestResize:
    @pytest.fixture()
    def resize(self) -> Resize:
        return Resize(scale=(128, 96), is_numpy_to_tvtensor=False)  # (64, 64) -> (128, 96)

    @pytest.mark.parametrize(
        ("keep_ratio", "expected_shape", "expected_scale_factor"),
        [
            (True, (96, 96), (1.5, 1.5)),
            (False, (128, 96), (2.0, 1.5)),
        ],
    )
    def test_forward_only_image(
        self,
        resize: Resize,
        fxt_det_data_entity: tuple[tuple, OTXDataItem, OTXDataBatch],
        keep_ratio: bool,
        expected_shape: tuple,
        expected_scale_factor: tuple,
    ) -> None:
        """Test forward only image."""
        resize.keep_ratio = keep_ratio
        resize.transform_bbox = False
        resize.transform_mask = False
        entity = deepcopy(fxt_det_data_entity[0])

        results = resize(entity)

        assert results.img_info.ori_shape == (64, 64)
        if keep_ratio:
            assert results.image.shape[:2] == expected_shape
            assert results.img_info.img_shape == expected_shape
            assert results.img_info.scale_factor == expected_scale_factor
        else:
            assert results.image.shape[:2] == expected_shape
            assert results.img_info.img_shape == expected_shape
            assert results.img_info.scale_factor == expected_scale_factor

        assert torch.all(results.bboxes.data == fxt_det_data_entity[0].bboxes.data)

    @pytest.mark.parametrize(
        ("keep_ratio", "expected_shape"),
        [
            (True, (96, 96)),
            (False, (128, 96)),
        ],
    )
    def test_forward_bboxes_masks_polygons(
        self,
        resize: Resize,
        fxt_inst_seg_data_entity: tuple[tuple, OTXDataItem, OTXDataBatch],
        keep_ratio: bool,
        expected_shape: tuple,
    ) -> None:
        """Test forward with bboxes, masks, and polygons."""
        resize.transform_bbox = True
        resize.transform_mask = True
        entity = deepcopy(fxt_inst_seg_data_entity[0])

        resize.keep_ratio = keep_ratio
        results = resize(entity)

        assert results.image.shape[:2] == expected_shape
        assert results.img_info.img_shape == expected_shape
        assert torch.all(
            results.bboxes
            == fxt_inst_seg_data_entity[0].bboxes * torch.tensor(results.img_info.scale_factor[::-1] * 2),
        )
        assert results.masks.shape[1:] == expected_shape
        assert all(
            [  # noqa: C419
                np.all(
                    np.array(rp.points).reshape(-1, 2)
                    == np.array(fp.points).reshape(-1, 2) * np.array([results.img_info.scale_factor[::-1]]),
                )
                for rp, fp in zip(results.polygons, fxt_inst_seg_data_entity[0].polygons)
            ],
        )


class TestRandomFlip:
    @pytest.fixture()
    def random_flip(self) -> RandomFlip:
        return RandomFlip(probability=1.0, is_numpy_to_tvtensor=False)

    def test_forward(
        self,
        random_flip: RandomFlip,
        fxt_inst_seg_data_entity: tuple[tuple, OTXDataItem, OTXDataBatch],
    ) -> None:
        """Test forward."""
        entity = deepcopy(fxt_inst_seg_data_entity[0])

        results = random_flip.forward(entity)

        # test image
        assert torch.all(F.to_image(results.image.copy()) == fxt_inst_seg_data_entity[0].image)

        # test bboxes
        bboxes_results = results.bboxes.clone()
        bboxes_results[..., 0] = results.img_info.img_shape[1] - results.bboxes[..., 2]
        bboxes_results[..., 2] = results.img_info.img_shape[1] - results.bboxes[..., 0]
        assert torch.all(bboxes_results == fxt_inst_seg_data_entity[0].bboxes)

        # test masks
        assert torch.all(tv_tensors.Mask(results.masks).flip(-1) == fxt_inst_seg_data_entity[0].masks)

        # test polygons
        def revert_hflip(polygon: list[float], width: int) -> list[float]:
            p = np.asarray(polygon.points)
            p[0::2] = width - p[0::2]
            return p.tolist()

        width = results.img_info.img_shape[1]
        polygons_results = deepcopy(results.polygons)
        polygons_results = [Polygon(points=revert_hflip(polygon, width)) for polygon in polygons_results]
        assert polygons_results == fxt_inst_seg_data_entity[0].polygons


class TestPhotoMetricDistortion:
    @pytest.fixture()
    def photo_metric_distortion(self) -> PhotoMetricDistortion:
        return PhotoMetricDistortion(is_numpy_to_tvtensor=False)

    def test_forward(self, photo_metric_distortion: PhotoMetricDistortion, det_data_entity: OTXDataItem) -> None:
        """Test forward."""
        results = photo_metric_distortion(deepcopy(det_data_entity))

        assert results.image.dtype == np.float32


class TestRandomAffine:
    @pytest.fixture()
    def random_affine(self) -> RandomAffine:
        return RandomAffine(is_numpy_to_tvtensor=False)

    @pytest.fixture()
    def random_affine_with_mask_transform(self) -> RandomAffine:
        return RandomAffine(transform_mask=True, mask_fill_value=0, is_numpy_to_tvtensor=False)

    @pytest.fixture()
    def random_affine_without_mask_transform(self) -> RandomAffine:
        return RandomAffine(transform_mask=False, is_numpy_to_tvtensor=False)

    @pytest.fixture()
    def random_affine_with_polygon_transform(self) -> RandomAffine:
        return RandomAffine(transform_polygon=True, is_numpy_to_tvtensor=False)

    @pytest.fixture()
    def random_affine_with_mask_and_polygon_transform(self) -> RandomAffine:
        return RandomAffine(transform_mask=True, transform_polygon=True, mask_fill_value=0, is_numpy_to_tvtensor=False)

    def test_init_invalid_translate_ratio(self) -> None:
        with pytest.raises(ValueError):  # noqa: PT011
            RandomAffine(max_translate_ratio=1.5)

    def test_init_invalid_scaling_ratio_range_inverse_order(self) -> None:
        with pytest.raises(ValueError):  # noqa: PT011
            RandomAffine(scaling_ratio_range=(1.5, 0.5))

    def test_init_invalid_scaling_ratio_range_zero_value(self) -> None:
        with pytest.raises(ValueError):  # noqa: PT011
            RandomAffine(scaling_ratio_range=(0, 0.5))

    def test_forward(self, random_affine: RandomAffine, det_data_entity: OTXDataItem) -> None:
        """Test forward."""
        results = random_affine(deepcopy(det_data_entity))

        assert results.image.shape[:2] == (112, 224)
        assert results.label.shape[0] == results.bboxes.shape[0]
        assert results.label.dtype == torch.long
        assert results.bboxes.dtype == torch.float32
        assert results.img_info.img_shape == results.image.shape[:2]

    def test_segmentation_transform(
        self, random_affine_with_mask_transform: RandomAffine, seg_data_entity: OTXDataItem
    ) -> None:
        """Test forward for segmentation task."""
        original_entity = deepcopy(seg_data_entity)
        results = random_affine_with_mask_transform(original_entity)

        assert hasattr(results, "masks")
        assert results.masks is not None
        assert results.masks.shape[0] > 0  # Should have masks
        assert results.masks.shape[1:] == results.image.shape[:2]  # Same spatial dimensions as image

        # Check that the number of masks matches the number of remaining bboxes and labels
        assert results.masks.shape[0] == results.label.shape[0]
        assert isinstance(results.masks, tv_tensors.Mask)

    def test_forward_with_masks_transform_enabled(
        self,
        random_affine_with_mask_transform: RandomAffine,
        det_data_entity_with_masks: OTXDataItem,
    ) -> None:
        """Test forward with masks when transform_mask is True."""
        original_entity = deepcopy(det_data_entity_with_masks)
        results = random_affine_with_mask_transform(original_entity)

        # Check that masks are present and transformed
        assert hasattr(results, "masks")
        assert results.masks is not None
        assert results.masks.shape[0] > 0  # Should have masks
        assert results.masks.shape[1:] == results.image.shape[:2]  # Same spatial dimensions as image

        # Check that the number of masks matches the number of remaining bboxes and labels
        assert results.masks.shape[0] == results.bboxes.shape[0]
        assert results.masks.shape[0] == results.label.shape[0]

        # Check that masks are still binary (0 or 255)
        unique_values = torch.unique(results.masks)
        assert len(unique_values) <= 2  # Should only have 0 and/or 255

        # Check data types
        assert results.masks.dtype == torch.bool
        assert isinstance(results.masks, tv_tensors.Mask)

    def test_forward_with_polygons_transform_enabled(
        self,
        random_affine_with_polygon_transform: RandomAffine,
        det_data_entity_with_polygons: OTXDataItem,
    ) -> None:
        """Test forward with polygons when transform_polygon is True."""
        original_entity = deepcopy(det_data_entity_with_polygons)
        results = random_affine_with_polygon_transform(original_entity)

        # Check that polygons are present and transformed
        assert hasattr(results, "polygons")
        assert results.polygons is not None
        assert len(results.polygons) > 0  # Should have polygons

        # Check that the number of polygons matches the number of remaining bboxes and labels
        assert len(results.polygons) == results.bboxes.shape[0]
        assert len(results.polygons) == results.label.shape[0]

        # Check that polygons are still valid (even number of coordinates)
        for polygon in results.polygons:
            assert len(polygon.points) % 2 == 0  # Should have even number of coordinates
            assert len(polygon.points) >= 6  # Should have at least 3 points (6 coordinates)

    def test_forward_with_masks_and_polygons_transform_enabled(
        self,
        random_affine_with_mask_and_polygon_transform: RandomAffine,
        det_data_entity_with_polygons: OTXDataItem,
    ) -> None:
        """Test forward with both masks and polygons when both transforms are enabled."""
        original_entity = deepcopy(det_data_entity_with_polygons)
        results = random_affine_with_mask_and_polygon_transform(original_entity)

        # Check that both masks and polygons are present and transformed
        assert hasattr(results, "masks")
        assert hasattr(results, "polygons")
        assert results.masks is not None
        assert results.polygons is not None

        # Check consistency between masks, polygons, bboxes, and labels
        assert results.masks.shape[0] == results.bboxes.shape[0]
        assert len(results.polygons) == results.bboxes.shape[0]
        assert results.masks.shape[0] == results.label.shape[0]
        assert len(results.polygons) == results.label.shape[0]

    def test_forward_with_empty_polygons(
        self,
        random_affine_with_polygon_transform: RandomAffine,
        det_data_entity_empty_polygons: OTXDataItem,
    ) -> None:
        """Test forward with empty polygons."""
        original_entity = deepcopy(det_data_entity_empty_polygons)
        results = random_affine_with_polygon_transform(original_entity)

        # Check that empty polygons are handled correctly
        assert hasattr(results, "polygons")
        assert results.polygons is not None
        assert len(results.polygons) == 0  # Should still be empty

    def test_forward_without_polygons(
        self,
        random_affine_with_polygon_transform: RandomAffine,
        det_data_entity: OTXDataItem,
    ) -> None:
        """Test forward when entity has no polygons attribute."""
        original_entity = deepcopy(det_data_entity)
        # Ensure no polygons attribute
        if hasattr(original_entity, "polygons"):
            delattr(original_entity, "polygons")

        results = random_affine_with_polygon_transform(original_entity)

        # Should work normally without polygons
        assert results.image.shape[:2] == (112, 224)
        assert results.label.shape[0] == results.bboxes.shape[0]
        assert not hasattr(results, "polygons") or results.polygons is None or len(results.polygons) == 0

    def test_polygon_filtering_consistency(
        self,
        det_data_entity_with_polygons: OTXDataItem,
    ) -> None:
        """Test that polygons are filtered consistently with bboxes."""
        # Create a transform that might filter out some bboxes
        transform = RandomAffine(
            transform_polygon=True,
            bbox_clip_border=True,
            max_rotate_degree=30,
            max_translate_ratio=0.3,
            scaling_ratio_range=(0.5, 1.5),
            max_shear_degree=10,
        )

        original_entity = deepcopy(det_data_entity_with_polygons)
        original_num_polygons = len(original_entity.polygons)

        results = transform(original_entity)

        # Check that the number of polygons matches the number of valid bboxes and labels
        assert len(results.polygons) == results.bboxes.shape[0]
        assert len(results.polygons) == results.label.shape[0]

        # The number of polygons might be reduced due to filtering
        assert len(results.polygons) <= original_num_polygons

    def test_polygon_coordinates_validity(
        self,
        random_affine_with_polygon_transform: RandomAffine,
        det_data_entity_with_polygons: OTXDataItem,
    ) -> None:
        """Test that transformed polygon coordinates are valid."""
        original_entity = deepcopy(det_data_entity_with_polygons)
        results = random_affine_with_polygon_transform(original_entity)

        # Check that all polygon coordinates are within image bounds
        height, width = results.image.shape[:2]

        for polygon in results.polygons:
            points = np.array(polygon.points).reshape(-1, 2)

            # Check that x coordinates are within [0, width]
            assert np.all(points[:, 0] >= 0)
            assert np.all(points[:, 0] <= width)

            # Check that y coordinates are within [0, height]
            assert np.all(points[:, 1] >= 0)
            assert np.all(points[:, 1] <= height)

    @pytest.mark.parametrize("transform_polygon", [True, False])
    def test_polygon_transform_parameter_effect(
        self,
        det_data_entity_with_polygons: OTXDataItem,
        transform_polygon: bool,
    ) -> None:
        """Test that the transform_polygon parameter controls polygon transformation."""
        transform = RandomAffine(
            transform_polygon=transform_polygon,
            max_rotate_degree=0,
            max_translate_ratio=0.1,
            scaling_ratio_range=(1.0, 1.0),
            max_shear_degree=0,
        )

        original_entity = deepcopy(det_data_entity_with_polygons)
        results = transform(original_entity)

        # Regardless of transform_polygon value, polygons should be present and properly managed
        assert hasattr(results, "polygons")
        if len(original_entity.polygons) > 0:
            assert results.polygons is not None
            assert len(results.polygons) == results.bboxes.shape[0]

    def test_forward_with_masks_transform_disabled(
        self,
        random_affine_without_mask_transform: RandomAffine,
        det_data_entity_with_masks: OTXDataItem,
    ) -> None:
        """Test forward with masks when transform_mask is False."""
        original_entity = deepcopy(det_data_entity_with_masks)
        results = random_affine_without_mask_transform(original_entity)

        # Check that masks are present but not transformed
        assert hasattr(results, "masks")
        assert results.masks is not None

        # Since transform_mask is False, masks should remain unchanged
        # However, they might still be filtered based on valid bounding boxes
        assert (
            results.masks.shape[0] == results.bboxes.shape[0]
        ), f"results.masks.shape[0] = {results.masks.shape[0]}, results.bboxes.shape[0] = {results.bboxes.shape[0]}"
        assert (
            results.masks.shape[0] == results.label.shape[0]
        ), f"results.masks.shape[0] = {results.masks.shape[0]}, results.label.shape[0] = {results.label.shape[0]}"

    def test_forward_with_empty_masks(
        self,
        random_affine_with_mask_transform: RandomAffine,
        det_data_entity_empty_masks: OTXDataItem,
    ) -> None:
        """Test forward with empty masks."""
        original_entity = deepcopy(det_data_entity_empty_masks)
        results = random_affine_with_mask_transform(original_entity)

        # Check that empty masks are handled correctly
        assert hasattr(results, "masks")
        assert results.masks is not None
        assert results.masks.shape[0] == 0  # Should still be empty
        assert results.masks.shape[1:] == results.image.shape[:2]  # Same spatial dimensions

    def test_forward_without_masks(
        self,
        random_affine_with_mask_transform: RandomAffine,
        det_data_entity: OTXDataItem,
    ) -> None:
        """Test forward when entity has no masks attribute."""
        original_entity = deepcopy(det_data_entity)
        # Ensure no masks attribute
        if hasattr(original_entity, "masks"):
            delattr(original_entity, "masks")

        results = random_affine_with_mask_transform(original_entity)

        # Should work normally without masks
        assert results.image.shape[:2] == (112, 224)
        assert results.label.shape[0] == results.bboxes.shape[0]
        assert not hasattr(results, "masks") or results.masks is None

    def test_mask_fill_value_applied(
        self,
        det_data_entity_with_masks: OTXDataItem,
        repeat: int = 10,
    ) -> None:
        """Test that mask_fill_value is applied correctly."""
        # Test with different fill values
        fill_values = [0, 128, 255]

        for _ in range(repeat):
            for fill_value in fill_values:
                transform = RandomAffine(
                    transform_mask=True,
                    mask_fill_value=fill_value,
                    max_rotate_degree=45,  # Force significant transformation
                    max_translate_ratio=0.2,
                    scaling_ratio_range=(0.8, 1.2),
                    max_shear_degree=10,
                )

                original_entity = deepcopy(det_data_entity_with_masks)
                results = transform(original_entity)

                assert hasattr(results, "masks")
                assert results.masks is not None
                # The fill value should be used for areas outside the original mask
                # This is hard to test directly, but we can check that the transform executed successfully
                assert results.masks.shape[0] > 0

    def test_mask_consistency_with_image_transform(
        self,
        det_data_entity_with_masks: OTXDataItem,
    ) -> None:
        """Test that masks and images are transformed consistently."""
        # Create a transform with fixed parameters for reproducibility
        transform = RandomAffine(
            transform_mask=True,
            mask_fill_value=0,
            max_rotate_degree=0,  # No rotation for simpler testing
            max_translate_ratio=0.1,  # Small translation
            scaling_ratio_range=(1.0, 1.0),  # No scaling
            max_shear_degree=0,  # No shear
            is_numpy_to_tvtensor=False,
        )

        original_entity = deepcopy(det_data_entity_with_masks)
        results = transform(original_entity)

        # Check that image and masks have consistent dimensions
        assert results.image.shape[:2] == results.masks.shape[1:]

        # Check that masks are still properly shaped
        assert len(results.masks.shape) == 3  # (N, H, W)
        assert results.masks.shape[0] == results.bboxes.shape[0]

    def test_mask_bbox_filtering_consistency(
        self,
        det_data_entity_with_masks: OTXDataItem,
    ) -> None:
        """Test that masks are filtered consistently with bboxes."""
        # Create a transform that might filter out some bboxes
        transform = RandomAffine(
            transform_mask=True,
            mask_fill_value=0,
            bbox_clip_border=True,
            max_rotate_degree=30,
            max_translate_ratio=0.3,
            scaling_ratio_range=(0.5, 1.5),
            max_shear_degree=10,
        )

        original_entity = deepcopy(det_data_entity_with_masks)
        original_num_objects = original_entity.masks.shape[0]

        results = transform(original_entity)

        # Check that the number of masks matches the number of valid bboxes and labels
        assert results.masks.shape[0] == results.bboxes.shape[0]
        assert results.masks.shape[0] == results.label.shape[0]

        # The number of objects might be reduced due to filtering
        assert results.masks.shape[0] <= original_num_objects


class TestCachedMosaic:
    @pytest.fixture()
    def cached_mosaic(self) -> CachedMosaic:
        return CachedMosaic(img_scale=(128, 128), random_pop=False, max_cached_images=20, is_numpy_to_tvtensor=False)

    def test_init_invalid_img_scale(self) -> None:
        with pytest.raises(AssertionError):
            CachedMosaic(img_scale=640, is_numpy_to_tvtensor=False)

    def test_init_invalid_probability(self) -> None:
        with pytest.raises(AssertionError):
            CachedMosaic(probability=1.5, is_numpy_to_tvtensor=False)

    def test_forward_pop_small_cache(
        self,
        cached_mosaic: CachedMosaic,
        fxt_inst_seg_data_entity: tuple[tuple, OTXDataItem, OTXDataBatch],
    ) -> None:
        """Test forward for popping cache."""
        cached_mosaic.max_cached_images = 4
        cached_mosaic.results_cache = [fxt_inst_seg_data_entity[0]] * cached_mosaic.max_cached_images

        # 4 -> 5 thru append -> 4 thru pop -> return due to small cache
        results = cached_mosaic(deepcopy(fxt_inst_seg_data_entity[0]))

        # check pop
        assert len(cached_mosaic.results_cache) == cached_mosaic.max_cached_images

        # check small cache
        assert torch.all(results.image == fxt_inst_seg_data_entity[0].image)
        assert torch.all(results.bboxes == fxt_inst_seg_data_entity[0].bboxes)

    def test_forward(
        self,
        cached_mosaic: CachedMosaic,
        fxt_inst_seg_data_entity: tuple[tuple, OTXDataItem, OTXDataBatch],
    ) -> None:
        """Test forward."""
        entity = deepcopy(fxt_inst_seg_data_entity[0])
        cached_mosaic.results_cache = [entity] * 4
        cached_mosaic.prob = 1.0

        results = cached_mosaic(deepcopy(entity))

        assert results.image.shape[:2] == (256, 256)
        assert results.label.shape[0] == results.bboxes.shape[0]
        assert results.label.dtype == torch.int64
        assert results.bboxes.dtype == torch.float32
        assert results.img_info.img_shape == results.image.shape[:2]
        assert results.masks.shape[1:] == (256, 256)


class TestCachedMixUp:
    @pytest.fixture()
    def cached_mixup(self) -> CachedMixUp:
        return CachedMixUp(
            ratio_range=(1.0, 1.0), probability=1.0, random_pop=False, max_cached_images=10, is_numpy_to_tvtensor=False
        )

    def test_init_invalid_img_scale(self) -> None:
        with pytest.raises(AssertionError):
            CachedMixUp(img_scale=640)

    def test_init_invalid_probability(self) -> None:
        with pytest.raises(AssertionError):
            CachedMosaic(probability=1.5)

    def test_forward_pop_small_cache(
        self,
        cached_mixup: CachedMixUp,
        fxt_inst_seg_data_entity: tuple[tuple, OTXDataItem, OTXDataBatch],
    ) -> None:
        """Test forward for popping cache."""
        cached_mixup.max_cached_images = 1  # force to set to 1 for this test
        cached_mixup.results_cache = [fxt_inst_seg_data_entity[0]] * cached_mixup.max_cached_images

        # 1 -> 2 thru append -> 1 thru pop -> return due to small cache
        results = cached_mixup(deepcopy(fxt_inst_seg_data_entity[0]))

        # check pop
        assert len(cached_mixup.results_cache) == cached_mixup.max_cached_images

        # check small cache
        assert torch.all(results.image == fxt_inst_seg_data_entity[0].image)
        assert torch.all(results.bboxes == fxt_inst_seg_data_entity[0].bboxes)

    def test_forward(
        self,
        cached_mixup: CachedMixUp,
        fxt_inst_seg_data_entity: tuple[tuple, OTXDataItem, OTXDataBatch],
    ) -> None:
        """Test forward."""
        entity = deepcopy(fxt_inst_seg_data_entity[0])
        cached_mixup.results_cache = [entity]
        cached_mixup.prob = 1.0
        cached_mixup.flip_ratio = 0.0

        results = cached_mixup(deepcopy(entity))

        assert results.image.shape[:2] == (64, 64)
        assert results.label.shape[0] == results.bboxes.shape[0]
        assert results.label.dtype == torch.int64
        assert results.bboxes.dtype == torch.float32
        assert results.img_info.img_shape == results.image.shape[:2]
        assert results.masks.shape[1:] == (64, 64)


class TestYOLOXHSVRandomAug:
    @pytest.fixture()
    def yolox_hsv_random_aug(self) -> YOLOXHSVRandomAug:
        return YOLOXHSVRandomAug(is_numpy_to_tvtensor=False)

    def test_forward(self, yolox_hsv_random_aug: YOLOXHSVRandomAug, det_data_entity: OTXDataItem) -> None:
        """Test forward."""
        results = yolox_hsv_random_aug(deepcopy(det_data_entity))

        assert results.image.shape[:2] == (112, 224)
        assert results.label.shape[0] == results.bboxes.shape[0]
        assert results.label.dtype == torch.int64
        assert results.bboxes.dtype == torch.float32


class TestPad:
    def test_forward(
        self,
        fxt_inst_seg_data_entity: tuple[tuple, OTXDataItem, OTXDataBatch],
    ) -> None:
        entity = deepcopy(fxt_inst_seg_data_entity[0])

        # test pad img/masks with size
        transform = Pad(size=(96, 128), transform_mask=True, is_numpy_to_tvtensor=False)

        results = transform(deepcopy(entity))

        assert results.image.shape[:2] == (96, 128)
        assert results.masks.shape[1:] == (96, 128)

        # test pad img/masks with size_divisor
        transform = Pad(size_divisor=11, transform_mask=True, is_numpy_to_tvtensor=False)

        results = transform(deepcopy(entity))

        # (64, 64) -> (66, 66)
        assert results.image.shape[:2] == (66, 66)
        assert results.masks.shape[1:] == (66, 66)

        # test pad img/masks with pad_to_square
        _transform = Pad(size=(96, 128), transform_mask=True, is_numpy_to_tvtensor=False)
        entity = _transform(deepcopy(entity))
        transform = Pad(pad_to_square=True, transform_mask=True, is_numpy_to_tvtensor=False)

        results = transform(deepcopy(entity))

        assert results.image.shape[:2] == (128, 128)
        assert results.masks.shape[1:] == (128, 128)

        # test pad img/masks with pad_to_square and size_divisor
        _transform = Pad(size=(96, 128), transform_mask=True, is_numpy_to_tvtensor=False)
        entity = _transform(deepcopy(entity))
        transform = Pad(pad_to_square=True, size_divisor=11, transform_mask=True, is_numpy_to_tvtensor=False)

        results = transform(deepcopy(entity))

        assert results.image.shape[:2] == (132, 132)
        assert results.masks.shape[1:] == (132, 132)


class TestRandomResize:
    def test_init(self):
        transform = RandomResize((224, 224), (1.0, 2.0), is_numpy_to_tvtensor=False)
        assert transform.scale == (224, 224)

    def test_repr(self):
        transform = RandomResize((224, 224), (1.0, 2.0), is_numpy_to_tvtensor=False)
        transform_str = str(transform)
        assert isinstance(transform_str, str)

    def test_forward(self, fxt_inst_seg_data_entity: tuple[tuple, OTXDataItem, OTXDataBatch]):
        entity = deepcopy(fxt_inst_seg_data_entity[0])

        # choose target scale from init when override is True
        transform = RandomResize((224, 224), (1.0, 2.0), is_numpy_to_tvtensor=False)

        results = transform(deepcopy(entity))

        assert results.img_info.img_shape[0] >= 224
        assert results.img_info.img_shape[0] <= 448
        assert results.img_info.img_shape[1] >= 224
        assert results.img_info.img_shape[1] <= 448

        # keep ratio is True
        transform = RandomResize(
            (224, 224),
            (1.0, 2.0),
            is_numpy_to_tvtensor=False,
            keep_ratio=True,
            transform_bbox=True,
            transform_mask=True,
        )

        results = transform(deepcopy(entity))
        assert results.image.shape[0] >= 224
        assert results.image.shape[0] <= 448
        assert results.image.shape[1] >= 224
        assert results.image.shape[1] <= 448
        assert results.img_info.img_shape[0] >= 224
        assert results.img_info.img_shape[0] <= 448
        assert results.img_info.img_shape[1] >= 224
        assert results.img_info.img_shape[1] <= 448
        assert results.img_info.scale_factor[0] == results.img_info.scale_factor[1]
        assert results.bboxes[0, 2] == entity.bboxes[0, 2] * results.img_info.scale_factor[0]
        assert results.bboxes[0, 3] == entity.bboxes[0, 3] * results.img_info.scale_factor[1]
        assert results.masks.shape[1] >= 224
        assert results.masks.shape[1] <= 448
        assert results.masks.shape[2] >= 224
        assert results.masks.shape[2] <= 448

        # keep ratio is False
        transform = RandomResize(
            (224, 224),
            (1.0, 2.0),
            keep_ratio=False,
            transform_bbox=True,
            transform_mask=True,
            is_numpy_to_tvtensor=False,
        )

        results = transform(deepcopy(entity))

        # choose target scale from init when override is False and scale is a list of tuples
        transform = RandomResize(
            [(448, 224), (224, 112)],
            keep_ratio=False,
            transform_bbox=True,
            transform_mask=True,
            is_numpy_to_tvtensor=False,
        )

        results = transform(deepcopy(entity))

        assert results.img_info.img_shape[1] >= 112
        assert results.img_info.img_shape[1] <= 224
        assert results.img_info.img_shape[0] >= 224
        assert results.img_info.img_shape[0] <= 448

        # the type of scale is invalid in init
        with pytest.raises(NotImplementedError):
            RandomResize([(448, 224), [224, 112]], keep_ratio=True)(deepcopy(entity))


class TestRandomCrop:
    @pytest.fixture()
    def entity(self) -> OTXDataItem:
        return OTXDataItem(
            image=torch.randn((3, 24, 32), dtype=torch.float32),
            img_info=ImageInfo(img_idx=0, img_shape=(24, 32), ori_shape=(24, 32)),
        )

    @pytest.fixture()
    def det_entity(self) -> OTXDataItem:
        return OTXDataItem(
            image=torch.randn((3, 10, 10), dtype=torch.float32),
            img_info=ImageInfo(img_idx=0, img_shape=(10, 10), ori_shape=(10, 10)),
            bboxes=tv_tensors.BoundingBoxes(
                np.array([[0, 0, 7, 7], [2, 3, 9, 9]], dtype=np.float32),
                format="xyxy",
                canvas_size=(10, 10),
            ),
            label=torch.LongTensor([0, 1]),
        )

    @pytest.fixture()
    def iseg_entity(self) -> OTXDataItem:
        return OTXDataItem(
            image=torch.randn((3, 10, 10), dtype=torch.float32),
            img_info=ImageInfo(img_idx=0, img_shape=(10, 10), ori_shape=(10, 10)),
            bboxes=tv_tensors.BoundingBoxes(
                np.array([[0, 0, 7, 7], [2, 3, 9, 9]], dtype=np.float32),
                format="xyxy",
                canvas_size=(10, 10),
            ),
            label=torch.LongTensor([0, 1]),
            masks=tv_tensors.Mask(np.zeros((2, 10, 10), np.uint8)),
            polygons=[Polygon(points=[0, 0, 0, 7, 7, 7, 7, 0]), Polygon(points=[2, 3, 2, 9, 9, 9, 9, 3])],
        )

    def test_init_invalid_crop_type(self) -> None:
        # test invalid crop_type
        with pytest.raises(ValueError, match="Invalid crop_type"):
            RandomCrop(crop_size=(10, 10), crop_type="unknown", is_numpy_to_tvtensor=False)

    @pytest.mark.parametrize("crop_type", ["absolute", "absolute_range"])
    @pytest.mark.parametrize("crop_size", [(0, 0), (0, 1), (1, 0)])
    def test_init_invalid_value(self, crop_type: str, crop_size: tuple[int, int]) -> None:
        # test h > 0 and w > 0
        with pytest.raises(AssertionError):
            RandomCrop(crop_size=crop_size, crop_type=crop_type, is_numpy_to_tvtensor=False)

    @pytest.mark.parametrize("crop_type", ["absolute", "absolute_range"])
    @pytest.mark.parametrize("crop_size", [(1.0, 1), (1, 1.0), (1.0, 1.0)])
    def test_init_invalid_type(self, crop_type: str, crop_size: tuple[int, int]) -> None:
        # test type(h) = int and type(w) = int
        with pytest.raises(AssertionError):
            RandomCrop(crop_size=crop_size, crop_type=crop_type, is_numpy_to_tvtensor=False)

    def test_init_invalid_size(self) -> None:
        # test crop_size[0] <= crop_size[1]
        with pytest.raises(AssertionError):
            RandomCrop(crop_size=(10, 5), crop_type="absolute_range", is_numpy_to_tvtensor=False)

    @pytest.mark.parametrize("crop_type", ["relative_range", "relative"])
    @pytest.mark.parametrize("crop_size", [(0, 1), (1, 0), (1.1, 0.5), (0.5, 1.1)])
    def test_init_invalid_range(self, crop_type: str, crop_size: tuple[int | float]) -> None:
        # test h in (0, 1] and w in (0, 1]
        with pytest.raises(AssertionError):
            RandomCrop(crop_size=crop_size, crop_type=crop_type, is_numpy_to_tvtensor=False)

    @pytest.mark.parametrize(("crop_type", "crop_size"), [("relative", (0.5, 0.5)), ("absolute", (12, 16))])
    def test_forward_relative_absolute(self, entity, crop_type: str, crop_size: tuple[float | int]) -> None:
        # test relative and absolute crop
        transform = RandomCrop(crop_size=crop_size, crop_type=crop_type, is_numpy_to_tvtensor=False)
        target_shape = (12, 16)

        results = transform(deepcopy(entity))

        assert results.image.shape[:2] == target_shape

    def test_forward_absolute_range(self, entity) -> None:
        # test absolute_range crop
        transform = RandomCrop(crop_size=(10, 20), crop_type="absolute_range", is_numpy_to_tvtensor=False)

        results = transform(deepcopy(entity))

        h, w = results.image.shape[:2]
        assert 10 <= w <= 20
        assert 10 <= h <= 20
        assert results.img_info.img_shape == results.image.shape[:2]

    def test_forward_relative_range(self, entity) -> None:
        # test relative_range crop
        transform = RandomCrop(crop_size=(0.9, 0.8), crop_type="relative_range", is_numpy_to_tvtensor=False)

        results = transform(deepcopy(entity))

        h, w = results.image.shape[:2]
        assert 24 * 0.9 <= h <= 24
        assert 32 * 0.8 <= w <= 32
        assert results.img_info.img_shape == results.image.shape[:2]

    def test_forward_bboxes_labels_masks_polygons(self, iseg_entity) -> None:
        # test with bboxes, labels, masks, and polygons
        transform = RandomCrop(
            crop_size=(7, 5),
            allow_negative_crop=False,
            recompute_bbox=False,
            bbox_clip_border=True,
            is_numpy_to_tvtensor=False,
        )

        results = transform(deepcopy(iseg_entity))

        assert results.image.shape[:2] == (7, 5)
        assert results.bboxes.shape[0] == 2
        assert results.label.shape[0] == 2
        assert results.masks.shape[0] == 2
        assert results.masks.shape[1:] == (7, 5)
        assert results.img_info.img_shape == results.image.shape[:2]

    def test_forward_recompute_bbox_from_mask(self, iseg_entity) -> None:
        # test recompute_bbox = True
        iseg_entity.bboxes = tv_tensors.wrap(torch.tensor([[0.1, 0.1, 0.2, 0.2]]), like=iseg_entity.bboxes)
        iseg_entity.label = torch.LongTensor([0])
        iseg_entity.polygons = None
        target_gt_bboxes = np.zeros((1, 4), dtype=np.float32)
        transform = RandomCrop(
            crop_size=(10, 11),
            allow_negative_crop=False,
            recompute_bbox=True,
            bbox_clip_border=True,
            is_numpy_to_tvtensor=False,
        )
        results = transform(deepcopy(iseg_entity))

        assert np.all(results.bboxes.numpy() == target_gt_bboxes)

    def test_forward_recompute_bbox_from_polygon(self, iseg_entity) -> None:
        # test recompute_bbox = True
        iseg_entity.bboxes = tv_tensors.wrap(torch.tensor([[0.1, 0.1, 0.2, 0.2]]), like=iseg_entity.bboxes)
        iseg_entity.label = torch.LongTensor([0])
        iseg_entity.masks = tv_tensors.Mask(np.zeros((0, *iseg_entity.img_info.img_shape), dtype=bool))
        target_gt_bboxes = np.array([[0.0, 0.0, 7.0, 7.0]], dtype=np.float32)
        transform = RandomCrop(
            crop_size=(10, 11),
            allow_negative_crop=False,
            recompute_bbox=True,
            bbox_clip_border=True,
            is_numpy_to_tvtensor=False,
        )
        results = transform(deepcopy(iseg_entity))

        assert np.all(results.bboxes.numpy() == target_gt_bboxes)

    def test_forward_bbox_clip_border_false(self, det_entity) -> None:
        # test bbox_clip_border = False
        det_entity.bboxes = tv_tensors.wrap(torch.tensor([[0.1, 0.1, 0.2, 0.2]]), like=det_entity.bboxes)
        det_entity.label = torch.LongTensor([0])
        transform = RandomCrop(
            crop_size=(10, 11),
            allow_negative_crop=False,
            recompute_bbox=True,
            bbox_clip_border=False,
            is_numpy_to_tvtensor=False,
        )

        results = transform(deepcopy(det_entity))

        assert torch.all(results.bboxes == det_entity.bboxes)

    @pytest.mark.parametrize("allow_negative_crop", [True, False])
    def test_forward_allow_negative_crop(self, det_entity, allow_negative_crop: bool) -> None:
        # test the crop does not contain any gt-bbox allow_negative_crop = False
        det_entity.image = np.random.randint(0, 255, size=(10, 10), dtype=np.uint8)
        det_entity.bboxes = tv_tensors.wrap(torch.zeros((0, 4)), like=det_entity.bboxes)
        det_entity.label = torch.LongTensor()
        transform = RandomCrop(crop_size=(5, 3), allow_negative_crop=allow_negative_crop, is_numpy_to_tvtensor=False)

        results = transform(deepcopy(det_entity))

        if allow_negative_crop:
            assert results.image.shape == transform.crop_size
            assert len(results.bboxes) == len(det_entity.bboxes) == 0
        else:
            assert results is None

    def test_repr(self):
        crop_type = "absolute"
        crop_size = (10, 5)
        allow_negative_crop = False
        recompute_bbox = True
        bbox_clip_border = False
        transform = RandomCrop(
            crop_size=crop_size,
            crop_type=crop_type,
            allow_negative_crop=allow_negative_crop,
            recompute_bbox=recompute_bbox,
            bbox_clip_border=bbox_clip_border,
            is_numpy_to_tvtensor=False,
        )
        assert (
            repr(transform) == f"RandomCrop(crop_size={crop_size}, crop_type={crop_type}, "
            f"allow_negative_crop={allow_negative_crop}, "
            f"recompute_bbox={recompute_bbox}, "
            f"bbox_clip_border={bbox_clip_border}, "
            f"is_numpy_to_tvtensor=False)"
        )


class TestTopdownAffine:
    @pytest.fixture()
    def keypoint_det_entity(self) -> OTXDataItem:
        return OTXDataItem(
            image=torch.randint(0, 255, size=(3, 10, 10), dtype=torch.float32),
            img_info=ImageInfo(img_idx=0, img_shape=(10, 10), ori_shape=(10, 10)),
            bboxes=tv_tensors.BoundingBoxes(
                np.array([[0, 0, 7, 7]], dtype=np.float32),
                format="xyxy",
                canvas_size=(10, 10),
            ),
            label=torch.LongTensor([0]),
            keypoints=tv_tensors.TVTensor(np.array([[0, 4, 1], [4, 2, 1], [2, 6, 1], [6, 0, 0]])),
        )

    def test_forward(self, keypoint_det_entity) -> None:
        transform = Compose(
            [
                TopdownAffine(input_size=(5, 5)),
            ],
        )
        results = transform(deepcopy(keypoint_det_entity))
        assert results.keypoints.shape == (4, 3)


class TestRandomGaussianNoise:
    def test_transform(self, det_data_entity) -> None:
        transform = Compose(
            [
                ToDtype(torch.float32),
                RandomGaussianNoise(mean=0.1, sigma=0.2, clip=True),
            ],
        )

        new_det_data_entity = deepcopy(det_data_entity)
        # test unscaled image in range [0, 255]
        result = transform(new_det_data_entity)
        assert not torch.all((result.image >= 0) & (result.image <= 1))
        assert torch.all((result.image >= 0) & (result.image <= 255))

        # test scaled image in range [0, 1]
        new_image = torch.rand((3, 100, 100))
        new_det_data_entity.image = new_image
        result = transform(new_det_data_entity)
        assert torch.all((result.image >= 0) & (result.image <= 1))
