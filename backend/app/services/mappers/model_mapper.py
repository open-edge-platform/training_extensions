# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from app.db.schema import ModelDB
from app.schemas.model import Model


class ModelMapper:
    """Mapper for Model schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(model_db: ModelDB) -> Model:
        """Convert Model db entity to schema."""

        return Model.model_validate(model_db, from_attributes=True)

    @staticmethod
    def from_schema(model: Model) -> ModelDB:
        """Convert Model schema to db model."""

        return ModelDB(**model.model_dump(mode="json"))
