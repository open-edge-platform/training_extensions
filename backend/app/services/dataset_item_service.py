# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from datetime import datetime
from io import BytesIO
from typing import BinaryIO
from uuid import UUID

from app.schemas.dataset_item import DatasetItem
from app.services.base import ResourceNotFoundError, ResourceType

logger = logging.getLogger(__name__)


class DatasetItemService:
    def create_dataset_item(self, file: BinaryIO) -> DatasetItem:
        """Creates a new dataset item"""
        raise NotImplementedError

    def list_dataset_items(
        self,
        limit: int,  # noqa: ARG002
        offset: int,  # noqa: ARG002
        start_date: datetime | None,  # noqa: ARG002
        end_date: datetime | None,  # noqa: ARG002
    ) -> list[DatasetItem]:
        """Get information about available dataset items"""
        return []

    def get_dataset_item_by_id(self, dataset_item_id: UUID) -> DatasetItem:
        """Get a dataset item by its ID"""
        raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))

    def get_dataset_item_binary_by_id(self, dataset_item_id: UUID) -> BytesIO:
        """Get a dataset item binary content by its ID"""
        raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))

    def get_dataset_item_thumbnail_by_id(self, dataset_item_id: UUID) -> BytesIO:
        """Get a dataset item thumbnail binary content by its ID"""
        raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))

    def delete_dataset_item(self, dataset_item_id: UUID) -> None:
        """Delete a dataset item by its ID"""
        raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))
