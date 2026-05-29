# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom collate function producing Ultralytics batch dicts."""

from __future__ import annotations

from typing import Any

import torch


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate adapter dicts into an Ultralytics-compatible batch.

    Stacks images into ``(B, C, H, W)`` and concatenates per-sample
    annotations with correct per-image ``batch_idx``.

    Instance masks are collated in **overlap format** (matching upstream
    Ultralytics ``overlap_mask=True``): a single ``(B, H, W)`` index map
    per batch where pixel values 1..N identify instance ownership.
    Instances are sorted by area descending so smaller masks overwrite
    larger ones in overlapping regions.
    """
    new_batch: dict[str, Any] = {}

    new_batch["img"] = torch.stack([b["img"] for b in batch], dim=0)

    all_cls: list[torch.Tensor] = []
    all_bboxes: list[torch.Tensor] = []
    all_batch_idx: list[torch.Tensor] = []

    for i, b in enumerate(batch):
        cls = torch.as_tensor(b["cls"], dtype=torch.float32)
        bboxes = torch.as_tensor(b["bboxes"], dtype=torch.float32)
        n = cls.shape[0]
        all_cls.append(cls)
        all_bboxes.append(bboxes)
        all_batch_idx.append(torch.full((n,), i, dtype=torch.float32))

    new_batch["cls"] = torch.cat(all_cls, dim=0) if all_cls else torch.zeros((0, 1), dtype=torch.float32)
    new_batch["bboxes"] = torch.cat(all_bboxes, dim=0) if all_bboxes else torch.zeros((0, 4), dtype=torch.float32)
    new_batch["batch_idx"] = torch.cat(all_batch_idx, dim=0) if all_batch_idx else torch.zeros(0, dtype=torch.float32)

    # Geometry fields (lists, one per image).
    new_batch["ori_shape"] = [b["ori_shape"] for b in batch]
    new_batch["resized_shape"] = [b["resized_shape"] for b in batch]
    new_batch["ratio_pad"] = [b["ratio_pad"] for b in batch]
    new_batch["im_file"] = [b.get("im_file", "") for b in batch]

    # Instance masks — overlap index map format (B, H, W).
    if "masks" in batch[0]:
        overlap_maps: list[torch.Tensor] = []
        sem_maps: list[torch.Tensor] = []

        for i, b in enumerate(batch):
            masks = b["masks"]
            if not isinstance(masks, torch.Tensor):
                masks = torch.as_tensor(masks, dtype=torch.float32)

            n_inst = masks.shape[0]
            if n_inst == 0:
                if masks.ndim == 3:
                    h, w = masks.shape[1], masks.shape[2]
                else:
                    h, w = 1, 1
                overlap_maps.append(torch.zeros((h, w), dtype=torch.uint8))
                sem_maps.append(torch.zeros((h, w), dtype=torch.float32))
                continue

            # Sort by area descending (larger first, smaller overwrite).
            areas = masks.float().sum(dim=(1, 2))  # (N,)
            sorted_idx = torch.argsort(areas, descending=True)
            masks = masks[sorted_idx]

            # Reorder cls/bboxes for this image to match sorted order.
            cls_i = all_cls[i]  # (N, 1)
            bboxes_i = all_bboxes[i]  # (N, 4)
            cls_i = cls_i[sorted_idx]
            bboxes_i = bboxes_i[sorted_idx]
            # Update the lists so that the concatenated batch reflects sorted order.
            all_cls[i] = cls_i
            all_bboxes[i] = bboxes_i

            # Build overlap index map: paint instances 1..N (smaller overwrite larger).
            h, w = masks.shape[1], masks.shape[2]
            index_map = torch.zeros((h, w), dtype=torch.uint8)
            for k in range(n_inst):
                index_map[masks[k] > 0] = k + 1
            overlap_maps.append(index_map)

            # Build sem_masks from index map + class labels.
            cls_flat = cls_i.squeeze(-1)  # (N,)
            sem = torch.zeros((h, w), dtype=torch.float32)
            fg = index_map > 0
            if fg.any():
                sem[fg] = cls_flat[(index_map[fg].long() - 1)]
            sem_maps.append(sem)

        new_batch["masks"] = torch.stack(overlap_maps, dim=0)  # (B, H, W)
        new_batch["sem_masks"] = torch.stack(sem_maps, dim=0)  # (B, H, W)

        # Rebuild cls/bboxes/batch_idx from reordered per-image lists.
        new_batch["cls"] = torch.cat(all_cls, dim=0) if all_cls else torch.zeros((0, 1), dtype=torch.float32)
        new_batch["bboxes"] = torch.cat(all_bboxes, dim=0) if all_bboxes else torch.zeros((0, 4), dtype=torch.float32)
        # batch_idx doesn't change (just counts per image).

    elif "sem_masks" in batch[0]:
        # Fallback: sem_masks without instance masks (shouldn't happen for seg).
        new_batch["sem_masks"] = torch.stack([b["sem_masks"] for b in batch], dim=0)

    return new_batch
