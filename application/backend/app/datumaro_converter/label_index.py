# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from uuid import UUID

from app.models import Label


class LabelIndex:
    """Knows the mapping between label IDs and their indices."""

    def __init__(self, labels: Sequence[Label]) -> None:
        self._id_to_index = {label.id: idx for idx, label in enumerate(labels)}
        self._labels = labels

    def get_index(self, label_id: UUID) -> int | None:
        """Returns the index for a label ID, or None if not found."""
        return self._id_to_index.get(label_id)

    def get_indices(self, label_ids: list[UUID]) -> list[int] | None:
        """Returns indices for multiple label IDs, or None if any not found."""
        indices = [self._id_to_index.get(lid) for lid in label_ids]
        return indices if None not in indices else None  # pyrefly: ignore

    @property
    def label_names(self) -> tuple[str, ...]:
        """Returns label names for category creation."""
        return tuple(label.name for label in self._labels)
