# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

from datumaro.experimental import Dataset
from datumaro.experimental.categories import LabelCategories
from loguru import logger

from app.datumaro_converter.factories import SampleFactory
from app.models import DatasetItem, Media

CONVERSION_BATCH_SIZE = 50


class DatasetConverter:
    """
    Knows how to convert dataset items to Datumaro format using a factory.

    The DatasetConverter orchestrates the conversion process by fetching dataset items in batches, delegating sample
    creation to a SampleFactory, and assembling the results into a Datumaro Dataset. It acts as a coordinator that
    knows the conversion workflow but delegates the specifics of sample creation to the injected factory.

    Attributes:
        _factory: The SampleFactory that knows how to create specific sample types.
        _get_items: Callback function to retrieve batches of dataset items and media.
        _get_path: Callback function to resolve media file paths for dataset items.
        _batch_size: Number of items to process in each batch.

    Example:
        >>> factory = DetectionSampleFactory(labels)
        >>> converter = DatasetConverter(
        ...     sample_factory=factory,
        ...     get_dataset_items_and_media=lambda offset, limit: fetch_items(offset, limit),
        ...     get_media_path=lambda item: f"/images/{item.id}.jpg",
        ...     batch_size=100
        ... )
        >>> dataset = converter.convert()
    """

    def __init__(
        self,
        sample_factory: SampleFactory,
        get_dataset_items_and_media: Callable[[int, int], list[tuple[DatasetItem, Media]]],
        get_media_path: Callable[[Media], str],
        batch_size: int = CONVERSION_BATCH_SIZE,
    ):
        self._factory = sample_factory
        self._get_items = get_dataset_items_and_media
        self._get_path = get_media_path
        self._batch_size = batch_size

    def convert(self) -> Dataset:
        """
        Converts the dataset using the configured factory.

        Orchestrates the entire conversion process by:
        1. Creating an empty Datumaro dataset with appropriate categories
        2. Iterating through dataset items in batches
        3. Converting each item to a sample using the factory
        4. Appending valid samples to the dataset

        Returns:
            A Datumaro Dataset containing all successfully converted samples with the appropriate sample type and
            label categories.

        Raises:
            Exception: If sample creation or dataset appending fails. The exception is logged with context
                (item ID, sample type) before being re-raised.
        """
        dataset = self._create_empty_dataset()

        for batch in self._iterate_batches():
            self._process_batch(dataset, batch)

        return dataset

    def _create_empty_dataset(self) -> Dataset:
        return Dataset(
            self._factory.sample_type,
            categories={"label": LabelCategories(labels=self._factory.label_index.label_names)},
        )

    def _iterate_batches(self):
        offset = 0
        while True:
            batch = self._get_items(offset, self._batch_size)
            if not batch:
                break
            yield batch
            offset += len(batch)

    def _process_batch(self, dataset: Dataset, batch: list[tuple[DatasetItem, Media]]):
        for dataset_item, media in batch:
            self._process_item(dataset, dataset_item, media)

    def _process_item(self, dataset: Dataset, dataset_item: DatasetItem, media: Media):
        media_path = self._get_path(media)

        try:
            sample = self._factory.create_sample(dataset_item, media, media_path)
        except Exception:
            logger.error("Failed conversion: item={}, type={}", dataset_item.id, self._factory.sample_type)
            raise

        if sample:
            try:
                dataset.append(sample)
            except Exception:
                logger.error("Failed to append the converted sample {} to the dataset {}", sample, dataset)
                raise
