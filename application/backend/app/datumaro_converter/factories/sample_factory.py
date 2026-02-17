# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Sequence

from datumaro.experimental import Sample

from app.datumaro_converter.label_index import LabelIndex
from app.models import DatasetItem, Label, Media


class SampleFactory(ABC):
    """Knows how to create a specific type of Datumaro sample from dataset items."""

    def __init__(self, project_labels: Sequence[Label]):
        self._label_index = LabelIndex(project_labels)

    @abstractmethod
    def create_sample(self, dataset_item: DatasetItem, media: Media, image_path: str) -> Sample | None:
        """Creates a sample from dataset item."""

    @property
    @abstractmethod
    def sample_type(self) -> type[Sample]:
        """Returns the sample type this factory produces."""

    @property
    def label_index(self) -> LabelIndex:
        return self._label_index
