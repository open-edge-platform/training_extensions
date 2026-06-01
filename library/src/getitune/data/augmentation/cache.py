# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightweight cache utilities for CachedMosaic and CachedMixUp transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from getitune.data.dataset.base import VisionDataset
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


class CacheableMixin:
    """Mixin for augmentations that maintain a sample cache (Mosaic, MixUp).

    Provides shared cache management: frozen/unfrozen modes, stochastic
    refresh, and pre-warming for multi-worker DataLoaders.

    Subclasses must set ``max_cached_images`` and ``random_pop`` before
    calling :meth:`_init_cache`.
    """

    results_cache: list[_CachedSample]
    max_cached_images: int
    random_pop: bool
    _cache_frozen: bool
    _refresh_rate: float

    def _init_cache(self) -> None:
        """Initialize cache state. Call from subclass ``__init__``."""
        self.results_cache = []
        self._cache_frozen = False
        self._refresh_rate = 0.05

    def freeze_cache(self, refresh_rate: float = 0.05) -> None:
        """Freeze the cache with slow stochastic refresh after pre-warming.

        When frozen, the cache is no longer updated via FIFO eviction on
        every forward call.  Instead, on each call, there is a small
        probability (``refresh_rate``) that one random cache entry is
        replaced with the current sample.  This keeps the cache slowly
        evolving with new data while maintaining the diversity of the
        pre-warmed pool.

        This is used with multi-worker DataLoaders (spawn context) where
        each worker receives a copy of the pre-warmed cache at spawn time.
        Without this mechanism, aggressive FIFO eviction causes each
        worker to independently fragment the cache.

        Args:
            refresh_rate: Probability of replacing one cache entry per
                forward call.  Lower values keep the pre-warmed pool
                stable longer; higher values bring in fresh samples
                faster.  Default 0.05 (~1 in 20 samples triggers a
                replacement).
        """
        self._cache_frozen = True
        self._refresh_rate = refresh_rate

    def pre_cache(self, dataset: VisionDataset) -> None:
        """Populate cache from dataset and freeze with stochastic refresh.

        Call before creating a multi-worker DataLoader so all workers
        inherit a full, diverse cache.  Iterating ``dataset[i]`` triggers
        the transform pipeline which populates ``results_cache`` via
        the subclass's ``forward()`` method.

        Args:
            dataset: The training dataset (must support ``len()`` and ``[]``).
        """
        n = min(self.max_cached_images, len(dataset))
        if len(self.results_cache) >= n:
            self.freeze_cache()
            return
        for i in range(n):
            dataset[i]
        if len(self.results_cache) < n:
            msg = (
                f"{type(self).__name__}.pre_cache: expected {n} cached samples "
                f"but got {len(self.results_cache)}. Ensure the cacheable transform "
                f"is part of the dataset's CPU augmentation pipeline."
            )
            raise RuntimeError(msg)
        self.freeze_cache()

    def _update_cache(self, sample: _CachedSample) -> None:
        """Append to cache (with eviction) or stochastically refresh an entry.

        Call from subclass ``forward()`` with a cloned sample.
        """
        if not self._cache_frozen:
            self.results_cache.append(sample)
            if len(self.results_cache) > self.max_cached_images:
                idx = int(torch.randint(0, len(self.results_cache), (1,)).item()) if self.random_pop else 0
                self.results_cache.pop(idx)
        elif self.results_cache and torch.rand(1).item() < self._refresh_rate:
            replace_idx = int(torch.randint(0, len(self.results_cache), (1,)).item())
            self.results_cache[replace_idx] = sample
