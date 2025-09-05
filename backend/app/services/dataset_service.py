# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from datetime import datetime
from pathlib import Path
from typing import BinaryIO
from uuid import UUID

from app.schemas.dataset_item import DatasetItem
from app.services.base import ResourceNotFoundError, ResourceType

logger = logging.getLogger(__name__)


class DatasetService:
    def create_dataset_item(self, project_id: UUID, file: BinaryIO) -> DatasetItem:
        """Creates a new dataset item"""
        raise NotImplementedError

    def list_dataset_items(
        self,
        project_id: UUID,  # noqa: ARG002
        limit: int = 20,  # noqa: ARG002
        offset: int = 0,  # noqa: ARG002
        start_date: datetime | None = None,  # noqa: ARG002
        end_date: datetime | None = None,  # noqa: ARG002
    ) -> list[DatasetItem]:
        """Get information about available dataset items"""
        return []

    def get_dataset_item_by_id(self, project_id: UUID, dataset_item_id: UUID) -> DatasetItem:  # noqa: ARG002
        """Get a dataset item by its ID"""
        raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))

    def get_dataset_item_binary_path_by_id(self, project_id: UUID, dataset_item_id: UUID) -> Path | str:  # noqa: ARG002
        """Get a dataset item binary content by its ID"""
        raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))

    def get_dataset_item_thumbnail_path_by_id(self, project_id: UUID, dataset_item_id: UUID) -> Path | str:  # noqa: ARG002
        """Get a dataset item thumbnail binary content by its ID"""
        raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))

    def delete_dataset_item(self, project_id: UUID, dataset_item_id: UUID) -> None:  # noqa: ARG002
        """Delete a dataset item by its ID"""
        raise ResourceNotFoundError(ResourceType.DATASET_ITEM, str(dataset_item_id))
