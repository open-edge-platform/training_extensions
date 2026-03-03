# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

from datumaro.experimental import Sample

from app.datumaro_converter.domain import LabelIndex
from app.models import DatasetItem, Label, Media

SampleT = TypeVar("SampleT", bound=Sample)


class SampleFactory(ABC, Generic[SampleT]):
    """Knows how to create a specific type of Datumaro sample from dataset items."""

    sample_type: type[SampleT]

    def __init__(self, project_labels: Sequence[Label]):
        self._label_index = LabelIndex(project_labels)

    @abstractmethod
    def create_sample(self, dataset_item: DatasetItem, media: Media, image_path: str) -> SampleT | None:
        """Creates a sample from dataset item."""
        ...

    @property
    def label_index(self) -> LabelIndex:
        return self._label_index
