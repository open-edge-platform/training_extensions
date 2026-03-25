# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

from datumaro.experimental import LazyImage, LazyVideoFrame, MediaInfo, Sample

from app.datumaro_converter.domain import LabelIndex, SampleMode
from app.models import DatasetItem, Image, Label, Media, VideoFrame

SampleT = TypeVar("SampleT", bound=Sample)


class SampleFactory(ABC, Generic[SampleT]):
    """Knows how to create a specific type of Datumaro sample from dataset items."""

    _sample_type_map: dict[SampleMode, type[SampleT]]

    def __init__(self, project_labels: Sequence[Label], mode: SampleMode):
        self._label_index = LabelIndex(project_labels)
        self._mode = mode
        self.sample_type = self._sample_type_map[mode]

    @abstractmethod
    def create_sample(self, dataset_item: DatasetItem, media: Media, media_path: str) -> SampleT | None:
        """Creates a sample from dataset item."""
        ...

    @property
    def label_index(self) -> LabelIndex:
        return self._label_index

    @staticmethod
    def _get_dm_media_with_info(item: Media, media_path: str) -> tuple[LazyImage | LazyVideoFrame, MediaInfo]:
        match item:
            case VideoFrame(frame_index=frame_index):
                media_item = LazyVideoFrame(video_path=media_path, frame_index=frame_index)
                return media_item, MediaInfo.from_media(media_item)
            case Image(width=width, height=height):
                media_item = LazyImage(media_path)
                return media_item, MediaInfo(width=width, height=height)
            case _:
                raise ValueError(f"Unsupported media type for Datumaro conversion: {item}")
