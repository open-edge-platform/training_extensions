# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom collate function producing Ultralytics batch dicts."""

from __future__ import annotations

from typing import Any

import torch


def ultralytics_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate adapter dicts into an Ultralytics-compatible batch.

    Stacks images into ``(B, C, H, W)`` and concatenates per-sample
    annotations with correct per-image ``batch_idx``.
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

    # Instance masks (segmentation).
    if "masks" in batch[0]:
        all_masks: list[torch.Tensor] = []
        for b in batch:
            masks = b["masks"]
            if not isinstance(masks, torch.Tensor):
                masks = torch.as_tensor(masks, dtype=torch.float32)
            all_masks.append(masks)
        new_batch["masks"] = torch.cat(all_masks, dim=0) if all_masks else torch.zeros((0, 1, 1), dtype=torch.float32)

    return new_batch
