import logging
import multiprocessing as mp
from multiprocessing.synchronize import Condition as ConditionClass
from threading import Thread

from app.db import get_db_session
from app.repositories import PipelineRepository, SinkRepository, SourceRepository
from app.schemas import DisconnectedSinkConfig, DisconnectedSourceConfig, Sink, Source
from app.services.mappers import SinkMapper, SourceMapper
from app.utils import Singleton

logger = logging.getLogger(__name__)


class ActivePipelineService(metaclass=Singleton):
    """
    A singleton service used in workers for loading pipeline-based application configuration from SQLite database.

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
                return

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
                logger.debug("Configuration changes detected. Process: %s", mp.current_process().name)
                self._load_app_config()

    def get_source_config(self) -> Source:
        return self._source

    def get_sink_config(self) -> Sink:
        return self._sink
