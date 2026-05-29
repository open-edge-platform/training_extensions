# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightweight cache utilities for CachedMosaic and CachedMixUp transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import torch

    from getitune.data.entity.sample import DetectionSample, InstanceSegmentationSample


class _CachedSample:
    """Lightweight cache entry storing only cloned tensor data for mosaic/mixup.

    Avoids the extreme cost of ``copy.deepcopy`` on full ``BaseSample``
    objects (which walk the entire Datumaro/PyTorch object graph).  Only
    the tensor data read during mosaic/mixup assembly is stored, cloned
    via fast ``Tensor.clone()`` (pure memcpy).
    """

    __slots__ = ("bboxes", "image", "label", "masks")

    def __init__(
        self,
        image: torch.Tensor,
        bboxes: torch.Tensor,
        label: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> None:
        self.image = image
        self.bboxes = bboxes
        self.label = label
        self.masks = masks


def _clone_for_cache(sample: DetectionSample | InstanceSegmentationSample) -> _CachedSample:
    """Create a lightweight cache entry with cloned tensor data.

    Cost: ~3ms for a 3x640x640 float32 image (memcpy only),
    vs. ~260ms for ``copy.deepcopy`` on a full BaseSample.
    """
    masks = getattr(sample, "masks", None)
    label = cast("torch.Tensor", sample.label)
    return _CachedSample(
        image=sample.image.clone(),
        bboxes=sample.bboxes.clone(),
        label=label.clone(),
        masks=masks.clone() if masks is not None else None,
    )
