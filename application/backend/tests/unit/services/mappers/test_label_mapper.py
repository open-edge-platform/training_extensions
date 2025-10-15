# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from uuid import uuid4

import pytest

from app.db.schema import LabelDB
from app.schemas import Label
from app.services.mappers import LabelMapper

SUPPORTED_LABEL_MAPPING = [
    (
        Label(
            name="cat",
            color="#FF5733",
            hotkey="C",
        ),
        LabelDB(name="cat", color="#FF5733", hotkey="C"),
    ),
    (
        Label(
            name="dog",
        ),  # type: ignore[call-arg]
        LabelDB(name="dog"),
    ),
]


class TestLabelMapper:
    """Test suite for LabelMapper methods."""

    @pytest.mark.parametrize("schema_instance,expected_db", SUPPORTED_LABEL_MAPPING.copy())
    def test_from_schema(self, schema_instance, expected_db):
        project_id = uuid4()
        expected_db.id = str(schema_instance.id)
        actual_db = LabelMapper.from_schema(project_id=project_id, label=schema_instance)
        assert actual_db.id == expected_db.id
        assert actual_db.project_id == str(project_id)
        assert actual_db.name == expected_db.name
        assert actual_db.color == expected_db.color
        assert actual_db.hotkey == expected_db.hotkey

    @pytest.mark.parametrize("db_instance,expected_schema", [(v, k) for (k, v) in SUPPORTED_LABEL_MAPPING.copy()])
    def test_to_schema(self, db_instance, expected_schema):
        db_instance.id = str(expected_schema.id)
        actual_schema = LabelMapper.to_schema(db_instance)
        assert actual_schema == expected_schema
