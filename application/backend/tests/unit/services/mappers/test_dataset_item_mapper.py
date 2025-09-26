# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest

from app.db.schema import DatasetItemDB
from app.schemas import DatasetItem
from app.schemas.dataset_item import DatasetItemFormat, DatasetItemSubset
from app.services.mappers import DatasetItemMapper

UUID0 = uuid4()

SUPPORTED_DATASET_ITEM_MAPPING = [
    (
        DatasetItem(
            name="DatasetItem1",
            format=DatasetItemFormat.JPG,
            width=1024,
            height=768,
            size=1024,
            source_id=UUID0,
            subset=DatasetItemSubset.TRAINING,
        ),
        DatasetItemDB(
            name="DatasetItem1",
            format="jpg",
            width=1024,
            height=768,
            size=1024,
            source_id=str(UUID0),
            subset="training",
        ),
    )
]


class TestDatasetItemMapper:
    """Test suite for DatasetItemMapper methods."""

    @pytest.mark.parametrize("schema_instance,expected_db", SUPPORTED_DATASET_ITEM_MAPPING.copy())
    def test_from_schema(self, schema_instance, expected_db):
        actual_db = DatasetItemMapper.from_schema(schema_instance)
        assert actual_db.project_id == expected_db.project_id
        assert actual_db.name == expected_db.name
        assert actual_db.format == expected_db.format
        assert actual_db.width == expected_db.width
        assert actual_db.height == expected_db.height
        assert actual_db.size == expected_db.size
        assert actual_db.source_id == expected_db.source_id
        assert actual_db.subset == expected_db.subset

    @pytest.mark.parametrize(
        "db_instance,expected_schema", [(v, k) for (k, v) in SUPPORTED_DATASET_ITEM_MAPPING.copy()]
    )
    def test_to_schema(self, db_instance, expected_schema):
        db_instance.id = str(expected_schema.id)
        actual_schema = DatasetItemMapper.to_schema(db_instance)
        assert actual_schema == expected_schema
