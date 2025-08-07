import logging
import multiprocessing as mp
from collections.abc import Callable
from multiprocessing.synchronize import Condition as ConditionClass
from threading import Thread
from uuid import UUID

from app.db import get_db_session
from app.db.schema import SinkDB, SourceDB
from app.repositories import PipelineRepository, SinkRepository, SourceRepository
from app.schemas import DisconnectedSinkConfig, DisconnectedSourceConfig, Sink, Source
from app.services.mappers.sink_mapper import SinkMapper
from app.services.mappers.source_mapper import SourceMapper
from app.utils import Singleton

logger = logging.getLogger(__name__)


class ConfigUpdateFromChildProcessError(Exception):
    """Exception raised when a child process tries to update the configuration of the parent process."""

    def __init__(self):
        super().__init__(
            "Attempted to update the configuration from a child process; only the parent process can update it."
        )


class ActivePipelineService(metaclass=Singleton):
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
        self._active_pipeline_id: str | None = None
        self._source: Source = DisconnectedSourceConfig()
        self._sink: Sink = DisconnectedSinkConfig()
        self._load_app_config()

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

    @staticmethod
    def __ensure_active_pipeline() -> None:
        with get_db_session() as db:
            pipeline_repo = PipelineRepository(db)
            active = pipeline_repo.get_active_pipeline()
            if not active:
                raise ValueError("No active pipeline")

    def _load_app_config(self) -> None:
        logger.info("Loading configuration from database")
        with get_db_session() as db:
            repo = PipelineRepository(db)

            pipeline = repo.get_active_pipeline()
            if pipeline is None:
                return

            self._active_pipeline_id = pipeline.id

            # Get the source for this pipeline
            if pipeline.source_id:
                source_repo = SourceRepository(db)
                source = source_repo.get_by_id(pipeline.source_id)
                if source is not None:
                    self._source = SourceMapper.to_schema(source)

            # Get the sink for this pipeline
            if pipeline.sink_id:
                sink_repo = SinkRepository(db)
                sink = sink_repo.get_by_id(pipeline.sink_id)
                if sink is not None:
                    self._sink = SinkMapper.to_schema(sink)

    def _reload_config_daemon_routine(self) -> None:
        """Daemon thread to reload the configuration file when it changes."""
        if self.config_changed_condition is None:
            raise RuntimeError("daemon thread initialized without config_changed_condition")
        while True:
            with self.config_changed_condition:
                notified = self.config_changed_condition.wait(timeout=3)
                if not notified:  # awakened before of timeout
                    continue
                self._load_app_config()

    def _notify_config_changed(self) -> None:
        """Notify child processes that the configuration has changed."""
        if self.config_changed_condition is None:
            raise RuntimeError("Attempt to notify uninitialized config_changed_condition")
        with self.config_changed_condition:
            logger.debug("Notifying other processes about configuration changes")
            self.config_changed_condition.notify_all()

    def get_source_config(self) -> Source:
        return self._source

    def get_sink_config(self) -> Sink:
        return self._sink

    def __set_config(self, config: SourceDB | SinkDB, on_success: Callable[[], None]) -> None:
        with get_db_session() as db:
            pipeline_repo = PipelineRepository(db)

            if isinstance(config, SourceDB):
                source_repo = SourceRepository(db)
                source_repo.save(config)
                if self._active_pipeline_id is not None:
                    pipeline_repo.update_source(self._active_pipeline_id, config.id)
                    self._source.id = UUID(config.id)
            elif isinstance(config, SinkDB):
                sink_repo = SinkRepository(db)
                sink_repo.save(config)
                if self._active_pipeline_id is not None:
                    pipeline_repo.update_sink(self._active_pipeline_id, config.id)
                    self._sink.id = UUID(config.id)
            else:
                raise TypeError(f"Unsupported config type: {type(config)}")

            db.commit()
            on_success()

    def set_source_config(self, source_config: Source) -> None:
        """Creating new source and attaching to the active pipeline."""
        self.__ensure_parent_process()
        self.__ensure_active_pipeline()
        source_db = SourceMapper.from_schema(source_config)
        self.__set_config(source_db, self._notify_config_changed)
        self._source = source_config

    def set_sink_config(self, sink_config: Sink) -> None:
        """Creating new sink and attaching to the active pipeline."""
        self.__ensure_parent_process()
        self.__ensure_active_pipeline()
        sink_db = SinkMapper.from_schema(sink_config)
        self.__set_config(sink_db, self._notify_config_changed)
        self._sink = sink_config
