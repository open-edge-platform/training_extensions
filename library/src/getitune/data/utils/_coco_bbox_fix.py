# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Temporary workaround for a Datumaro COCO import bug.

Datumaro's ``_collect_instances_for_image`` can split multi-polygon COCO
annotations into one bounding box per polygon part. For annotations that
describe one object with several disjoint visible regions, that inflates the
ground-truth count and produces fragment boxes instead of a single box covering
the full object.

This module patches the Datumaro helper so each annotation yields one bounding
box computed from the union of all polygon parts. It also deduplicates exact
duplicate annotations to avoid counting the same object multiple times.

Remove this module once Datumaro ships a native fix.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from datumaro.experimental.data_formats.coco.helpers import _segmentation_to_poly_parts
import datumaro.experimental.data_formats.coco.helpers as coco_helpers

logger = logging.getLogger(__name__)

_ORIGINAL_FN: Any = None


def _fixed_collect_instances_for_image(
    image_id: int,
    instances_by_image: dict[int, list[dict]],
    cat_id_to_idx: dict[int, int],
) -> tuple[list[list[float] | None], list[np.ndarray], list[int | None], list[float], list[bool]]:
    """Return one bbox per annotation instead of one bbox per polygon part."""

    bboxes_list: list[list[float] | None] = []
    polygons_list: list[np.ndarray] = []
    labels_list: list[int | None] = []
    areas_list: list[float] = []
    iscrowd_list: list[bool] = []

    for ann in instances_by_image.get(image_id, []):
        cat_id = ann.get("category_id")
        category_idx = cat_id_to_idx.get(cat_id) if cat_id is not None else None
        segmentation = ann.get("segmentation")
        original_bbox = ann.get("bbox")
        original_area = ann.get("area")

        polygon_parts = _segmentation_to_poly_parts(segmentation)

        ic = ann.get("iscrowd", 0)
        try:
            iscrowd_val = bool(int(ic))
        except Exception:
            iscrowd_val = False

        valid_parts = [part for part in polygon_parts if part.size > 0]

        if valid_parts:
            all_points = np.concatenate(valid_parts, axis=0)
            x1 = float(all_points[:, 0].min())
            y1 = float(all_points[:, 1].min())
            x2 = float(all_points[:, 0].max())
            y2 = float(all_points[:, 1].max())
            computed_bbox: list[float] = [x1, y1, x2 - x1, y2 - y1]
            area = (x2 - x1) * (y2 - y1)
            combined_poly = all_points
        elif original_bbox is not None and len(original_bbox) == 4:
            computed_bbox = [float(v) for v in original_bbox]
            area = float(original_area) if original_area is not None else computed_bbox[2] * computed_bbox[3]
            combined_poly = np.zeros((0, 2), dtype=np.float32)
        else:
            computed_bbox = [0.0, 0.0, 0.0, 0.0]
            area = 0.0
            combined_poly = np.zeros((0, 2), dtype=np.float32)

        bboxes_list.append(computed_bbox)
        polygons_list.append(combined_poly)
        labels_list.append(category_idx)
        areas_list.append(area)
        iscrowd_list.append(iscrowd_val)

    if bboxes_list:
        seen: set[tuple[int | None, tuple[float, ...]]] = set()
        dedup_indices: list[int] = []
        for index, (bbox, label) in enumerate(zip(bboxes_list, labels_list)):
            key = (label, tuple(round(value, 2) for value in bbox) if bbox else ())
            if key not in seen:
                seen.add(key)
                dedup_indices.append(index)

        if len(dedup_indices) < len(bboxes_list):
            bboxes_list = [bboxes_list[index] for index in dedup_indices]
            polygons_list = [polygons_list[index] for index in dedup_indices]
            labels_list = [labels_list[index] for index in dedup_indices]
            areas_list = [areas_list[index] for index in dedup_indices]
            iscrowd_list = [iscrowd_list[index] for index in dedup_indices]

    return bboxes_list, polygons_list, labels_list, areas_list, iscrowd_list


def apply_coco_bbox_fix() -> None:
    """Patch Datumaro's COCO loader once per process."""
    global _ORIGINAL_FN  # noqa: PLW0603

    if _ORIGINAL_FN is not None:
        return


    _ORIGINAL_FN = coco_helpers._collect_instances_for_image  # noqa: SLF001
    coco_helpers._collect_instances_for_image = _fixed_collect_instances_for_image  # noqa: SLF001
    logger.debug("Applied Datumaro COCO multi-polygon bbox fix")


def revert_coco_bbox_fix() -> None:
    """Restore the original Datumaro helper, primarily for tests."""
    global _ORIGINAL_FN  # noqa: PLW0603

    if _ORIGINAL_FN is None:
        return

    import datumaro.experimental.data_formats.coco.helpers as coco_helpers

    coco_helpers._collect_instances_for_image = _ORIGINAL_FN  # noqa: SLF001
    _ORIGINAL_FN = None
    logger.debug("Reverted Datumaro COCO multi-polygon bbox fix")
