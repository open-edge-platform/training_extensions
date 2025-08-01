import logging
import multiprocessing as mp
from collections.abc import Callable
from multiprocessing.synchronize import Condition as ConditionClass
from threading import Thread

from app.db import get_db_session
from app.db.schema import SinkDB, SourceDB
from app.repositories import PipelineRepository
from app.repositories.sink_repo import SinkRepository
from app.repositories.source_repo import SourceRepository
from app.schemas.configuration import AppConfig, Sink, Source
from app.services.mappers.sink_mapper import SinkMapper
from app.services.mappers.source_mapper import SourceMapper
from app.utils.singleton import Singleton

logger = logging.getLogger(__name__)


class ConfigUpdateFromChildProcessError(Exception):
    """Exception raised when a child process tries to update the configuration of the parent process."""

    def __init__(self):
        super().__init__(
            "Attempted to update the configuration from a child process; only the parent process can update it."
        )


class ConfigurationService(metaclass=Singleton):
    """
    A singleton service for managing pipeline-based application configuration from SQLite database.

    This service handles loading, saving, and monitoring configuration changes based on the active pipeline.
    The configuration is built from Source -> Pipeline -> Sinks relationships.
    In multiprocess environments, only the parent process can modify the configuration,
    while child processes automatically reload configuration when changes are detected.

    Args:
        config_changed_condition: Multiprocessing Condition object for notifying child
                                processes of configuration changes. Required for child processes.

    Raises:
        ConfigUpdateFromChildProcessError: When a child process attempts to update configuration.
        ValueError: When config_changed_condition is None in a child process.
    """

    def __init__(self, config_changed_condition: ConditionClass | None = None) -> None:
        self.config_changed_condition = config_changed_condition
        self._source_mapper = SourceMapper()
        self._sink_mapper = SinkMapper()
        self._active_pipeline_id: str | None = None
        self._source_id: str | None = None
        self._sink_id: str | None = None
        self._app_config = self._load_app_config()

        # For child processes, start a daemon to monitor configuration changes and reload it when necessary.
        if not self.__is_parent_process():
            if self.config_changed_condition is None:
                raise ValueError("config_changed_condition must be provided for child processes")
            self._config_reload_daemon = Thread(
                target=self._reload_config_daemon_routine, name="Config reloader", daemon=True
            )
            self._config_reload_daemon.start()

    @staticmethod
    def __is_parent_process() -> bool:
        return mp.parent_process() is None

    def __ensure_parent_process(self) -> None:
        if not self.__is_parent_process():
            raise ConfigUpdateFromChildProcessError

    def _load_app_config(self) -> AppConfig:
        logger.info("Loading configuration from database")
        app_config = AppConfig()
        with get_db_session() as db:
            repo = PipelineRepository(db)

            pipeline = repo.get_active_pipeline()
            if pipeline is None:
                return AppConfig()

            self._active_pipeline_id = pipeline.id

            # Get the source for this pipeline
            if pipeline.source_id:
                source_repo = SourceRepository(db)
                source = source_repo.get_by_id(pipeline.source_id)
                if source is not None:
                    self._source_id = source.id
                    app_config.input = self._source_mapper.to_schema(source)

            # Get the sink for this pipeline
            if pipeline.sink_id:
                sink_repo = SinkRepository(db)
                sink = sink_repo.get_by_id(pipeline.sink_id)
                if sink is not None:
                    self._sink_id = sink.id
                    app_config.output = self._sink_mapper.to_schema(sink)

            return app_config

    def _reload_config_daemon_routine(self) -> None:
        """Daemon thread to reload the configuration file when it changes."""
        if self.config_changed_condition is None:
            raise RuntimeError("daemon thread initialized without config_changed_condition")
        while True:
            with self.config_changed_condition:
                notified = self.config_changed_condition.wait(timeout=3)
                if not notified:  # awakened before of timeout
                    continue
                self._app_config = self._load_app_config()

    def _notify_config_changed(self) -> None:
        """Notify child processes that the configuration has changed."""
        if self.config_changed_condition is None:
            raise RuntimeError("Attempt to notify uninitialized config_changed_condition")
        with self.config_changed_condition:
            logger.debug("Notifying other processes about configuration changes")
            self.config_changed_condition.notify_all()

    def get_app_config(self) -> AppConfig:
        return self._app_config

    def get_source_config(self) -> Source:
        return self._app_config.input

    def get_sink_config(self) -> Sink:
        return self._app_config.output

    def __set_config(self, config: SourceDB | SinkDB, on_success: Callable[[], None]) -> None:
        with get_db_session() as db:
            pipeline_repo = PipelineRepository(db)

            if isinstance(config, SourceDB):
                source_repo = SourceRepository(db)
                source_repo.save(config)
                if self._active_pipeline_id is not None:
                    pipeline_repo.update_source(self._active_pipeline_id, config.id)
            elif isinstance(config, SinkDB):
                sink_repo = SinkRepository(db)
                sink_repo.save(config)
                if self._active_pipeline_id is not None:
                    pipeline_repo.update_sink(self._active_pipeline_id, config.id)
            else:
                raise TypeError(f"Unsupported config type: {type(config)}")

            db.commit()
            on_success()

    def set_source_config(self, source_config: Source) -> None:
        """Creating new source and attaching to the active pipeline."""
        self.__ensure_parent_process()
        self._app_config.input = source_config
        source_db = self._source_mapper.from_schema(source_config)
        self.__set_config(source_db, self._notify_config_changed)

    def set_sink_config(self, sink_config: Sink) -> None:
        """Creating new sink and attaching to the active pipeline."""
        self.__ensure_parent_process()
        self._app_config.output = sink_config
        sink_db = self._sink_mapper.from_schema(sink_config)
        self.__set_config(sink_db, self._notify_config_changed)
