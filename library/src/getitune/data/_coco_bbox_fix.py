# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Temporary workaround for a Datumaro COCO import bug.

Datumaro's ``_collect_instances_for_image`` splits multi-polygon COCO
annotations into **one bounding box per polygon part**.  For annotations
that describe an occluded object with several disjoint visible regions
(e.g. a grape cluster partially hidden behind leaves), this produces many
tiny fragment boxes instead of a single box enclosing the whole object.

The fix below monkey-patches the function so that each COCO annotation
always yields exactly **one** bounding box computed from the union of all
its polygon parts, matching the original ``bbox`` semantics of the COCO
format.

Additionally, exact-duplicate annotations (same category + same bbox)
are deduplicated to avoid inflating the ground truth count, which
artificially caps recall and depresses mAP.

Remove this module once Datumaro ships a native fix.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_ORIGINAL_FN: Any = None


def _fixed_collect_instances_for_image(
    image_id: int,
    instances_by_image: dict[int, list[dict]],
    cat_id_to_idx: dict[int, int],
) -> tuple[list[list[float] | None], list[np.ndarray], list[int | None], list[float], list[bool]]:
    """Fixed version that emits one bbox per annotation, not one per polygon part."""
    # Keep local import to reuse the polygon-splitting helper unchanged.
    from datumaro.experimental.data_formats.coco.helpers import (
        _segmentation_to_poly_parts,
    )

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

        # Collect all valid polygon parts for this annotation.
        valid_parts = [p for p in polygon_parts if p.size > 0]

        if valid_parts:
            # Compute a single union bbox from ALL polygon parts.
            all_points = np.concatenate(valid_parts, axis=0)
            x1 = float(all_points[:, 0].min())
            y1 = float(all_points[:, 1].min())
            x2 = float(all_points[:, 0].max())
            y2 = float(all_points[:, 1].max())
            computed_bbox: list[float] = [x1, y1, x2 - x1, y2 - y1]
            area = (x2 - x1) * (y2 - y1)

            # Concatenate polygon parts into a single (N, 2) array so the
            # polygons field stays populated (useful for instance-seg masks).
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

    # Deduplicate exact-duplicate annotations (same label + same bbox).
    # Some COCO datasets contain redundant annotations that inflate the
    # ground truth count and artificially depress recall / mAP.
    if bboxes_list:
        seen: set[tuple[int | None, tuple[float, ...]]] = set()
        dedup_indices: list[int] = []
        for i, (bbox, label) in enumerate(zip(bboxes_list, labels_list)):
            key = (label, tuple(round(v, 2) for v in bbox) if bbox else ())
            if key not in seen:
                seen.add(key)
                dedup_indices.append(i)
        if len(dedup_indices) < len(bboxes_list):
            bboxes_list = [bboxes_list[i] for i in dedup_indices]
            polygons_list = [polygons_list[i] for i in dedup_indices]
            labels_list = [labels_list[i] for i in dedup_indices]
            areas_list = [areas_list[i] for i in dedup_indices]
            iscrowd_list = [iscrowd_list[i] for i in dedup_indices]

    return bboxes_list, polygons_list, labels_list, areas_list, iscrowd_list


def apply_coco_bbox_fix() -> None:
    """Monkey-patch Datumaro to fix multi-polygon bbox expansion.

    Safe to call multiple times — only patches once.
    """
    global _ORIGINAL_FN  # noqa: PLW0603

    import datumaro.experimental.data_formats.coco.helpers as coco_helpers

    if _ORIGINAL_FN is not None:
        return  # already patched

    _ORIGINAL_FN = coco_helpers._collect_instances_for_image  # noqa: SLF001
    coco_helpers._collect_instances_for_image = _fixed_collect_instances_for_image  # noqa: SLF001
    logger.debug("Applied Datumaro COCO multi-polygon bbox fix")


def revert_coco_bbox_fix() -> None:
    """Revert the monkey-patch (primarily for testing)."""
    global _ORIGINAL_FN  # noqa: PLW0603

    if _ORIGINAL_FN is None:
        return

    import datumaro.experimental.data_formats.coco.helpers as coco_helpers

    coco_helpers._collect_instances_for_image = _ORIGINAL_FN  # noqa: SLF001
    _ORIGINAL_FN = None
    logger.debug("Reverted Datumaro COCO multi-polygon bbox fix")
