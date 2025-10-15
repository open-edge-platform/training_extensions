# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from uuid import UUID

from app.db.schema import LabelDB
from app.schemas.label import Label


class LabelMapper:
    """Mapper for Label schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(label_db: LabelDB) -> Label:
        """Convert Label db entity to schema."""

        return Label.model_validate(label_db, from_attributes=True)

    @staticmethod
    def from_schema(project_id: UUID, label: Label) -> LabelDB:
        """Convert Label schema to db model."""

        label_db = LabelDB(**label.model_dump(mode="json"))
        label_db.project_id = str(project_id)
        return label_db
