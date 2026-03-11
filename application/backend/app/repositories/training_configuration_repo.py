# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.schema import TrainingConfigurationDB
from app.repositories.base import BaseRepository


class TrainingConfigurationRepository(BaseRepository[TrainingConfigurationDB]):
    def __init__(self, db: Session) -> None:
        super().__init__(db, TrainingConfigurationDB)

    def get_task_level_configuration(
        self,
        project_id: str,
    ) -> TrainingConfigurationDB | None:
        """
        Get task-level training configuration by project.

        Args:
            project_id: The ID of the project.

        Returns:
            The task-level training configuration if found, otherwise None.
        """
        stmt = select(TrainingConfigurationDB).where(
            TrainingConfigurationDB.project_id == project_id,
            TrainingConfigurationDB.model_architecture_id.is_(None),
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def get_algo_level_configuration(
        self,
        project_id: str,
        model_architecture_id: str,
    ) -> TrainingConfigurationDB | None:
        """
        Get algo-level training configuration by project and model architecture.

        Args:
            project_id: The ID of the project.
            model_architecture_id: The ID of the model architecture.

        Returns:
            The algo-level training configuration if found, otherwise None.
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
        Create or update the task/algo-level training configuration for a given project.

        If model_architecture_id is None, the configuration is considered task-level; otherwise, it's algo-level.

        Args:
            project_id: The ID of the project.
            model_architecture_id: The ID of the model architecture, or None for task-level configuration.
            configuration_data: The configuration data to store.

        Returns:
            TrainingConfigurationDB: The created or updated training configuration.
        """
        if model_architecture_id is None:
            existing_config = self.get_task_level_configuration(project_id=project_id)
        else:
            existing_config = self.get_algo_level_configuration(
                project_id=project_id,
                model_architecture_id=model_architecture_id,
            )

        if existing_config:
            existing_config.configuration_data = configuration_data
            self.save(existing_config)
            return existing_config

        new_config = TrainingConfigurationDB(
            project_id=project_id,
            model_architecture_id=model_architecture_id,
            configuration_data=configuration_data,
        )
        self.save(new_config)
        return new_config
