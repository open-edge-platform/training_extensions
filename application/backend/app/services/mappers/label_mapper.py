# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from app.db.schema import LabelDB
from app.schemas.label import LabelView


class LabelMapper:
    """Mapper for Label schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(label_db: LabelDB) -> LabelView:
        """Convert Label db entity to schema."""

        return LabelView.model_validate(label_db, from_attributes=True)
