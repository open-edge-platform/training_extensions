# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from app.db.schema import LabelDB
from app.schemas import LabelView
from app.services.mappers import LabelMapper

SUPPORTED_LABEL_MAPPING = [
    (
        LabelView(
            name="cat",
            color="#FF5733",
            hotkey="C",
        ),
        LabelDB(name="cat", color="#FF5733", hotkey="C"),
    ),
    (
        LabelView(
            name="dog",
            color="#FF5766",
        ),  # type: ignore[call-arg]
        LabelDB(name="dog", color="#FF5766"),
    ),
]


class TestLabelMapper:
    """Test suite for LabelMapper methods."""

    @pytest.mark.parametrize("db_instance,expected_schema", [(v, k) for (k, v) in SUPPORTED_LABEL_MAPPING.copy()])
    def test_to_schema(self, db_instance, expected_schema):
        db_instance.id = str(expected_schema.id)
        actual_schema = LabelMapper.to_schema(db_instance)
        assert actual_schema == expected_schema
