# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from app.db.schema import ModelRevisionDB
from app.schemas.model import Model, TrainingInfo, TrainingStatus


# TODO: add unit tests for this mapper when service layer is implemented
class ModelRevisionMapper:
    """Mapper for Model schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(model_db: ModelRevisionDB) -> Model:
        """Convert Model db entity to schema."""

        training_info = TrainingInfo(
            status=TrainingStatus(model_db.training_status),
            start_time=model_db.training_started_at,
            end_time=model_db.training_finished_at,
            dataset_revision_id=UUID(model_db.training_dataset_id) if model_db.training_dataset_id else None,
            label_schema_revision=model_db.label_schema_revision,
            configuration=model_db.training_configuration,
        )

        return Model(
            id=UUID(model_db.id),
            architecture=model_db.architecture,
            parent_revision=UUID(model_db.parent_revision) if model_db.parent_revision else None,
            training_info=training_info,
            files_deleted=model_db.files_deleted,
        )

    @staticmethod
    def from_schema(model: Model) -> ModelRevisionDB:
        """Convert Model schema to db model."""

        raise NotImplementedError
