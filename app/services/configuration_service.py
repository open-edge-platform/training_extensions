import logging
import multiprocessing as mp
from multiprocessing.synchronize import Condition as ConditionClass
from threading import Thread

import yaml
from pydantic import ValidationError

from app.schemas.configuration import AppConfig, InputConfig, OutputConfig
from app.utils.singleton import Singleton

logger = logging.getLogger(__name__)


class ConfigUpdateFromChildProcessError(Exception):
    """Exception raised when a child process tries to update the configuration of the parent process."""

    def __init__(self):
        super().__init__(
            "Attempted to update the configuration from a child process; only the parent process can update it."
        )


class ConfigurationService(metaclass=Singleton):
    def __init__(self, config_changed_condition: ConditionClass | None = None) -> None:
        self.config_changed_condition = config_changed_condition
        self._config_path = "config.yaml"
        self._app_config = self._load_app_config(self._config_path)

        # For child processes, start a daemon to monitor configuration changes and reload it when necessary.
        if not self.__is_parent_process():
            if self.config_changed_condition is None:
                raise ValueError("config_changed_condition must be provided for child processes")
            self._config_reload_daemon = Thread(
                target=self._reload_config_daemon_routine, name="Config reloader", daemon=True
            )
            self._config_reload_daemon.start()

    def __is_parent_process(self) -> bool:
        return mp.parent_process() is None

    def __ensure_parent_process(self) -> None:
        if not self.__is_parent_process():
            raise ConfigUpdateFromChildProcessError

    @staticmethod
    def _load_app_config(path: str) -> AppConfig:
        logger.info("Loading configuration from %s", path)
        with open(path) as f:
            raw_config = yaml.safe_load(f)
        try:
            return AppConfig(**raw_config)
        except ValidationError:
            logger.exception("Failed to parse configuration")
            raise

    def _reload_config_daemon_routine(self) -> None:
        """Daemon thread to reload the configuration file when it changes."""
        if self.config_changed_condition is None:
            raise RuntimeError("daemon thread initialized without config_changed_condition")
        while True:
            with self.config_changed_condition:
                notified = self.config_changed_condition.wait(timeout=3)
                if not notified:  # awakened before of timeout
                    continue
                self._app_config = self._load_app_config(self._config_path)

    def _notify_config_changed(self) -> None:
        """Notify child processes that the configuration has changed."""
        if self.config_changed_condition is None:
            raise RuntimeError("Attempt to notify uninitialized config_changed_condition")
        with self.config_changed_condition:
            logger.debug("Notifying other processes about configuration changes")
            self.config_changed_condition.notify_all()

    @staticmethod
    def _dump_app_config(app_config: AppConfig, path: str) -> None:
        """Dump the AppConfig to a YAML file."""
        with open(path, "w") as f:
            serialized_config = app_config.model_dump(mode="json")
            yaml.safe_dump(serialized_config, f, default_flow_style=False)
        logger.info("Configuration dumped to %s", path)

    def _save_config(self, app_config: AppConfig) -> None:
        """Save the current configuration to the file."""
        self._dump_app_config(app_config=app_config, path=self._config_path)
        self._notify_config_changed()

    def get_app_config(self) -> AppConfig:
        return self._app_config

    def get_input_config(self) -> InputConfig:
        return self._app_config.input

    def get_output_config(self) -> list[OutputConfig]:
        return self._app_config.outputs

    def set_input_config(self, input_config: InputConfig) -> None:
        self.__ensure_parent_process()
        self._app_config.input = input_config
        self._save_config(app_config=self._app_config)

    def set_output_config(self, outputs: list[OutputConfig]) -> None:
        self.__ensure_parent_process()
        self._app_config.outputs = outputs
        self._save_config(app_config=self._app_config)
