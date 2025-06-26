import logging

import yaml
from pydantic import ValidationError

from app.schemas.configuration import AppConfig, InputConfig, OutputConfig
from app.utils.singleton import Singleton

logger = logging.getLogger(__name__)


class ConfigurationService(metaclass=Singleton):
    def __init__(self) -> None:
        self._app_config = self._load_app_config("config.yaml")

    @staticmethod
    def _load_app_config(path: str) -> AppConfig:
        with open(path) as f:
            raw_config = yaml.safe_load(f)
        try:
            return AppConfig(**raw_config)
        except ValidationError:
            logger.exception("Failed to parse configuration")
            raise

    def get_app_config(self) -> AppConfig:
        return self._app_config

    def get_input_config(self) -> InputConfig:
        return self._app_config.input

    def get_output_config(self) -> OutputConfig:
        return self._app_config.output
