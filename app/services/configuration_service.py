import logging
import multiprocessing as mp
from collections.abc import Callable
from datetime import datetime
from multiprocessing.synchronize import Condition as ConditionClass
from threading import Thread

from sqlalchemy.orm import Session

from app.db.database import get_db_session
from app.db.schema import PipelineDB, SinkDB, SourceDB
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
        self._active_pipeline_id: int | None = None
        self._source_id: int | None = None
        self._sink_id: int | None = None
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

    @staticmethod
    def _get_active_pipeline(db: Session) -> PipelineDB | None:
        """Get the active pipeline from database."""
        return db.query(PipelineDB).filter(PipelineDB.is_running).first()

    def _load_app_config(self) -> AppConfig:
        logger.info("Loading configuration from database")
        app_config = AppConfig()
        with get_db_session() as db:
            # Get the active pipeline
            pipeline = self._get_active_pipeline(db)

            if pipeline is None:
                raise ValueError("No active pipeline found")

            self._active_pipeline_id = pipeline.id

            # Get the source for this pipeline
            if pipeline.source_id:
                source = db.query(SourceDB).filter(SourceDB.id == pipeline.source_id).first()
                self._source_id = source.id
                app_config.input = self._source_mapper.to_schema(source)

            # Get the sink for this pipeline
            if pipeline.sink_id:
                sink = db.query(SinkDB).filter(SinkDB.id == pipeline.sink_id).first()
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
            try:
                config.updated_at = datetime.now()
                db.add(config)
                db.flush()
                pipeline_db = db.query(PipelineDB).filter(PipelineDB.id == self._active_pipeline_id).first()
                if pipeline_db is None:
                    raise ValueError(f"Active pipeline with ID {self._active_pipeline_id} not found")

                if isinstance(config, SourceDB):
                    pipeline_db.source_id = config.id
                elif isinstance(config, SinkDB):
                    pipeline_db.sink_id = config.id
                else:
                    raise TypeError(f"Unsupported config type: {type(config)}")

                pipeline_db.updated_at = datetime.now()
                db.commit()
                on_success()
            except Exception:
                logger.exception("Failed to update configuration")
                db.rollback()
                raise

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
