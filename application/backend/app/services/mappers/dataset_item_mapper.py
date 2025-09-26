# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from app.db.schema import DatasetItemDB
from app.schemas.dataset_item import DatasetItem


class DatasetItemMapper:
    """Mapper for DatasetItem schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(dataset_item_db: DatasetItemDB) -> DatasetItem:
        """Convert DatasetItem db entity to schema."""

        return DatasetItem.model_validate(dataset_item_db, from_attributes=True)

    @staticmethod
    def from_schema(dataset_item: DatasetItem) -> DatasetItemDB:
        """Convert Model schema to db model."""

        return DatasetItemDB(**dataset_item.model_dump(mode="json"))
