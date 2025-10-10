# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import multiprocessing as mp
from multiprocessing.synchronize import Condition as ConditionClass
from threading import Thread
from uuid import UUID

from app.db import get_db_session
from app.repositories import PipelineRepository
from app.schemas import DisconnectedSinkConfig, DisconnectedSourceConfig, ProjectView, Sink, Source
from app.schemas.pipeline import DataCollectionPolicy

from .label_service import LabelService
from .mappers import ProjectMapper, SinkMapper, SourceMapper

logger = logging.getLogger(__name__)


class ActivePipelineService:
    """
    A service used in workers for loading pipeline-based application configuration from SQLite database.

    This service handles loading and monitoring configuration changes based on the active pipeline.
    The configuration is built from Source -> Pipeline -> Sinks relationships.

    Args:
        config_changed_condition: Multiprocessing Condition object for getting configuration updates in child
                                processes. Required for child processes.

    Raises:
        ValueError: When config_changed_condition is None in a child process.
    """

    def __init__(self, config_changed_condition: ConditionClass | None = None) -> None:
        self.config_changed_condition = config_changed_condition
        self._source: Source = DisconnectedSourceConfig()
        self._sink: Sink = DisconnectedSinkConfig()
        self._project: ProjectView | None = None
        self._data_collection_policies: list[DataCollectionPolicy] = []
        self._load_app_config()

        # For child processes, start a daemon to monitor configuration changes and reload it when necessary.
        if mp.parent_process() is not None:
            if self.config_changed_condition is None:
                raise ValueError("config_changed_condition must be provided for child processes")
            self._config_reload_daemon = Thread(
                target=self._reload_config_daemon_routine, name="Config reloader", daemon=True
            )
            self._config_reload_daemon.start()

    def reload(self) -> None:
        """Reload the application configuration from the database."""
        self._load_app_config()

    def _load_app_config(self) -> None:
        logger.info("Loading configuration from database")
        with get_db_session() as db:
            repo = PipelineRepository(db)

            # Loads the first active pipeline
            pipeline = repo.get_active_pipeline()
            if pipeline is None:
                self._source = DisconnectedSourceConfig()
                self._sink = DisconnectedSinkConfig()
                self._project = None
                self._data_collection_policies = []
                return

            source = pipeline.source
            if source is not None:
                self._source = SourceMapper.to_schema(source)

            sink = pipeline.sink
            if sink is not None:
                self._sink = SinkMapper.to_schema(sink)

            project = pipeline.project
            if project is not None:
                self._project = ProjectMapper.to_schema(project, LabelService(db).list_all(UUID(project.id)))
            self._data_collection_policies = [
                policy
                for policy in [
                    DataCollectionPolicy.model_validate(policy) for policy in pipeline.data_collection_policies
                ]
                if policy.enabled
            ]

    def _reload_config_daemon_routine(self) -> None:
        """Daemon thread to reload the configuration file when it changes."""
        if self.config_changed_condition is None:
            raise RuntimeError("daemon thread initialized without config_changed_condition")
        while True:
            with self.config_changed_condition:
                notified = self.config_changed_condition.wait(timeout=3)
                if not notified:  # awakened before of timeout
                    continue
                logger.debug("Configuration changes detected. Process: %s", mp.current_process().name)
                self._load_app_config()

    def get_source_config(self) -> Source:
        return self._source

    def get_sink_config(self) -> Sink:
        return self._sink

    def get_project(self) -> ProjectView | None:
        return self._project

    def get_data_collection_policies(self) -> list[DataCollectionPolicy]:
        return self._data_collection_policies
