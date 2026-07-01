# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Datumaro COCO multi-polygon bbox fix."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from getitune.data._coco_bbox_fix import (
    _fixed_collect_instances_for_image,
    apply_coco_bbox_fix,
    revert_coco_bbox_fix,
)


@pytest.fixture(autouse=True)
def _clean_patch() -> Iterator[None]:
    """Ensure the monkey-patch is reverted after each test."""
    yield
    revert_coco_bbox_fix()


class TestFixedCollectInstancesForImage:
    """Tests for the fixed _collect_instances_for_image function."""

    def _make_instances(self, annotations: list[dict]) -> dict[int, list[dict]]:
        """Helper to build instances_by_image dict."""
        return {1: annotations}

    def test_single_polygon_unchanged(self):
        """Single-polygon annotation should produce one bbox, same as original."""
        ann = {
            "category_id": 1,
            "segmentation": [[10.0, 20.0, 50.0, 20.0, 50.0, 60.0, 10.0, 60.0]],
            "bbox": [10, 20, 40, 40],
            "area": 1600.0,
        }
        instances = self._make_instances([ann])
        cat_map = {1: 0}

        bboxes, polygons, labels, areas, iscrowd = _fixed_collect_instances_for_image(1, instances, cat_map)

        assert len(bboxes) == 1
        assert bboxes[0] == pytest.approx([10.0, 20.0, 40.0, 40.0])
        assert labels == [0]
        assert areas[0] == pytest.approx(1600.0)
        assert iscrowd == [False]

    def test_multi_polygon_produces_single_union_bbox(self):
        """Multi-polygon annotation must produce ONE union bbox, not one per part."""
        ann = {
            "category_id": 2,
            "segmentation": [
                # Part 1: small box at top-left
                [10.0, 10.0, 20.0, 10.0, 20.0, 20.0, 10.0, 20.0],
                # Part 2: small box at bottom-right
                [80.0, 80.0, 100.0, 80.0, 100.0, 100.0, 80.0, 100.0],
            ],
            "bbox": [10, 10, 90, 90],
            "area": 8100.0,
        }
        instances = self._make_instances([ann])
        cat_map = {2: 1}

        bboxes, polygons, labels, areas, iscrowd = _fixed_collect_instances_for_image(1, instances, cat_map)

        assert len(bboxes) == 1, f"Expected 1 bbox but got {len(bboxes)}"
        # Union bbox: x1=10, y1=10, x2=100, y2=100 → xywh = [10, 10, 90, 90]
        assert bboxes[0] == pytest.approx([10.0, 10.0, 90.0, 90.0])
        assert labels == [1]
        assert areas[0] == pytest.approx(8100.0)

    def test_six_polygon_parts_still_one_bbox(self):
        """Annotation with 6 disjoint polygon parts (occluded object) → one bbox."""
        parts = [
            [5.0, 5.0, 15.0, 5.0, 15.0, 15.0, 5.0, 15.0],
            [30.0, 30.0, 40.0, 30.0, 40.0, 40.0, 30.0, 40.0],
            [60.0, 10.0, 70.0, 10.0, 70.0, 20.0, 60.0, 20.0],
            [1.0, 50.0, 4.0, 50.0, 4.0, 51.0, 1.0, 51.0],
            [90.0, 90.0, 95.0, 90.0, 95.0, 95.0, 90.0, 95.0],
            [45.0, 45.0, 55.0, 45.0, 55.0, 55.0, 45.0, 55.0],
        ]
        ann = {
            "category_id": 3,
            "segmentation": parts,
            "bbox": [1, 5, 94, 90],
            "area": 8460.0,
        }
        instances = self._make_instances([ann])
        cat_map = {3: 2}

        bboxes, polygons, labels, areas, iscrowd = _fixed_collect_instances_for_image(1, instances, cat_map)

        assert len(bboxes) == 1
        # Union: x_min=1, y_min=5, x_max=95, y_max=95 → xywh = [1, 5, 94, 90]
        assert bboxes[0] == pytest.approx([1.0, 5.0, 94.0, 90.0])
        assert labels == [2]

    def test_multiple_annotations_each_get_one_bbox(self):
        """Two annotations (one multi-polygon, one single) → two bboxes total."""
        ann1 = {
            "category_id": 1,
            "segmentation": [
                [10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0],
                [50.0, 50.0, 70.0, 50.0, 70.0, 70.0, 50.0, 70.0],
            ],
            "bbox": [10, 10, 60, 60],
            "area": 3600.0,
        }
        ann2 = {
            "category_id": 2,
            "segmentation": [[100.0, 100.0, 200.0, 100.0, 200.0, 200.0, 100.0, 200.0]],
            "bbox": [100, 100, 100, 100],
            "area": 10000.0,
        }
        instances = self._make_instances([ann1, ann2])
        cat_map = {1: 0, 2: 1}

        bboxes, polygons, labels, areas, iscrowd = _fixed_collect_instances_for_image(1, instances, cat_map)

        assert len(bboxes) == 2
        assert bboxes[0] == pytest.approx([10.0, 10.0, 60.0, 60.0])
        assert bboxes[1] == pytest.approx([100.0, 100.0, 100.0, 100.0])
        assert labels == [0, 1]

    def test_no_segmentation_uses_original_bbox(self):
        """Annotation without segmentation falls back to original bbox field."""
        ann = {
            "category_id": 1,
            "segmentation": None,
            "bbox": [5, 10, 20, 30],
            "area": 600.0,
        }
        instances = self._make_instances([ann])
        cat_map = {1: 0}

        bboxes, polygons, labels, areas, iscrowd = _fixed_collect_instances_for_image(1, instances, cat_map)

        assert len(bboxes) == 1
        assert bboxes[0] == pytest.approx([5.0, 10.0, 20.0, 30.0])
        assert areas[0] == pytest.approx(600.0)

    def test_empty_image_returns_empty(self):
        """Image with no annotations returns empty lists."""
        instances: dict[int, list[dict]] = {}
        cat_map = {1: 0}

        bboxes, polygons, labels, areas, iscrowd = _fixed_collect_instances_for_image(99, instances, cat_map)

        assert len(bboxes) == 0
        assert len(polygons) == 0
        assert len(labels) == 0

    def test_iscrowd_propagated(self):
        """iscrowd flag is correctly propagated."""
        ann = {
            "category_id": 1,
            "segmentation": [[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]],
            "bbox": [0, 0, 10, 10],
            "area": 100.0,
            "iscrowd": 1,
        }
        instances = self._make_instances([ann])
        cat_map = {1: 0}

        _, _, _, _, iscrowd = _fixed_collect_instances_for_image(1, instances, cat_map)

        assert iscrowd == [True]

    def test_rle_segmentation_handled(self):
        """RLE segmentation (dict) should produce empty polygon with fallback bbox."""
        ann = {
            "category_id": 1,
            "segmentation": {"counts": [0, 10, 5, 10], "size": [100, 100]},
            "bbox": [20, 30, 40, 50],
            "area": 2000.0,
        }
        instances = self._make_instances([ann])
        cat_map = {1: 0}

        bboxes, polygons, labels, areas, iscrowd = _fixed_collect_instances_for_image(1, instances, cat_map)

        assert len(bboxes) == 1
        assert bboxes[0] == pytest.approx([20.0, 30.0, 40.0, 50.0])
        assert polygons[0].shape == (0, 2)

    def test_polygon_parts_concatenated(self):
        """Multi-polygon parts should be concatenated in the polygons output."""
        ann = {
            "category_id": 1,
            "segmentation": [
                [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0],  # 4 points
                [20.0, 20.0, 30.0, 20.0, 30.0, 30.0, 20.0, 30.0],  # 4 points
            ],
            "bbox": [0, 0, 30, 30],
            "area": 900.0,
        }
        instances = self._make_instances([ann])
        cat_map = {1: 0}

        _, polygons, _, _, _ = _fixed_collect_instances_for_image(1, instances, cat_map)

        assert len(polygons) == 1
        # 4 points from part 1 + 4 points from part 2 = 8 points
        assert polygons[0].shape == (8, 2)

    def test_exact_duplicate_annotations_deduplicated(self):
        """Exact-duplicate annotations (same label + same bbox) should be collapsed to one."""
        ann = {
            "category_id": 1,
            "segmentation": [[10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]],
            "bbox": [10, 10, 40, 40],
            "area": 1600.0,
        }
        instances = self._make_instances([ann, ann])
        cat_map = {1: 0}

        bboxes, _, labels, _, _ = _fixed_collect_instances_for_image(1, instances, cat_map)

        assert len(bboxes) == 1
        assert labels == [0]

    def test_different_category_duplicates_kept(self):
        """Same bbox but different category should NOT be deduplicated."""
        ann1 = {
            "category_id": 1,
            "segmentation": [[10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]],
            "bbox": [10, 10, 40, 40],
            "area": 1600.0,
        }
        ann2 = {
            "category_id": 2,
            "segmentation": [[10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]],
            "bbox": [10, 10, 40, 40],
            "area": 1600.0,
        }
        instances = self._make_instances([ann1, ann2])
        cat_map = {1: 0, 2: 1}

        bboxes, _, labels, _, _ = _fixed_collect_instances_for_image(1, instances, cat_map)

        assert len(bboxes) == 2
        assert labels == [0, 1]

    def test_different_bbox_same_category_kept(self):
        """Same category but different bbox should NOT be deduplicated."""
        ann1 = {
            "category_id": 1,
            "segmentation": [[10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]],
            "bbox": [10, 10, 40, 40],
            "area": 1600.0,
        }
        ann2 = {
            "category_id": 1,
            "segmentation": [[60.0, 60.0, 100.0, 60.0, 100.0, 100.0, 60.0, 100.0]],
            "bbox": [60, 60, 40, 40],
            "area": 1600.0,
        }
        instances = self._make_instances([ann1, ann2])
        cat_map = {1: 0}

        bboxes, _, labels, _, _ = _fixed_collect_instances_for_image(1, instances, cat_map)

        assert len(bboxes) == 2
        assert labels == [0, 0]

    def test_triple_duplicates_collapsed(self):
        """Three identical annotations should be collapsed to one."""
        ann = {
            "category_id": 1,
            "segmentation": [[10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]],
            "bbox": [10, 10, 40, 40],
            "area": 1600.0,
        }
        instances = self._make_instances([ann, ann, ann])
        cat_map = {1: 0}

        bboxes, _, labels, _, _ = _fixed_collect_instances_for_image(1, instances, cat_map)

        assert len(bboxes) == 1
        assert labels == [0]


class TestPatchMechanism:
    """Tests for the apply/revert patch functions."""

    def test_apply_patches_function(self):
        """apply_coco_bbox_fix should replace the Datumaro function."""
        import datumaro.experimental.data_formats.coco.helpers as coco_helpers

        original = coco_helpers._collect_instances_for_image
        apply_coco_bbox_fix()

        assert coco_helpers._collect_instances_for_image is not original
        assert coco_helpers._collect_instances_for_image.__name__ == "_fixed_collect_instances_for_image"

    def test_revert_restores_original(self):
        """revert_coco_bbox_fix should restore the original function."""
        import datumaro.experimental.data_formats.coco.helpers as coco_helpers

        original = coco_helpers._collect_instances_for_image
        apply_coco_bbox_fix()
        revert_coco_bbox_fix()

        assert coco_helpers._collect_instances_for_image is original

    def test_idempotent_apply(self):
        """Calling apply twice should not break anything."""
        import datumaro.experimental.data_formats.coco.helpers as coco_helpers

        apply_coco_bbox_fix()
        first_patched = coco_helpers._collect_instances_for_image
        apply_coco_bbox_fix()

        assert coco_helpers._collect_instances_for_image is first_patched

    def test_revert_without_apply_is_noop(self):
        """Reverting without applying should do nothing."""
        import datumaro.experimental.data_formats.coco.helpers as coco_helpers

        original = coco_helpers._collect_instances_for_image
        revert_coco_bbox_fix()  # should not raise

        assert coco_helpers._collect_instances_for_image is original
