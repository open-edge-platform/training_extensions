# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.schema import TrainingConfigurationDB
from app.repositories.base import BaseRepository


class TrainingConfigurationRepository(BaseRepository[TrainingConfigurationDB]):
    def __init__(self, db: Session) -> None:
        super().__init__(db, TrainingConfigurationDB)

    def get_by_project_and_model_architecture(
        self,
        project_id: str,
        model_architecture_id: str | None = None,
    ) -> TrainingConfigurationDB | None:
        """
        Get training configuration by project ID and optional model architecture ID.

        Args:
            project_id (str): The ID of the project.
            model_architecture_id (str | None): The ID of the model architecture.
                If None, the operation targets the project-level configuration.
        """
        stmt = select(TrainingConfigurationDB).where(
            TrainingConfigurationDB.project_id == project_id,
            TrainingConfigurationDB.model_architecture_id == model_architecture_id,
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def create_or_update(
        self,
        project_id: str,
        model_architecture_id: str | None,
        configuration_data: dict,
    ) -> TrainingConfigurationDB:
        """
        Create or update a training configuration.

        If a configuration for the given project and model architecture exists, it is updated.
        Otherwise, a new configuration is created.

        Args:
            project_id (str): The ID of the project.
            model_architecture_id (str | None): The ID of the model architecture.
            configuration_data (dict): The configuration data to store.

        Returns:
            TrainingConfigurationDB: The created or updated training configuration.
        """
        existing = self.get_by_project_and_model_architecture(
            project_id=project_id,
            model_architecture_id=model_architecture_id,
        )

        if existing:
            existing.configuration_data = configuration_data
            self.save(existing)
            return existing

        new_config = TrainingConfigurationDB(
            project_id=project_id,
            model_architecture_id=model_architecture_id,
            configuration_data=configuration_data,
        )
        self.save(new_config)
        return new_config
