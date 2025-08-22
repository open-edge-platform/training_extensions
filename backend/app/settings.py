# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Application configuration management"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    # Application
    app_name: str = "Geti Tune"
    version: str = "0.1.0"
    summary: str = "Geti Tune server"
    description: str = (
        "Geti Tune allows to fine-tune computer vision models at the edge. "
        "It is a lightweight application that covers the complete model AI lifecycle, "
        "including advanced features for inference, model monitoring and data collection."
    )
    openapi_url: str = "/api/openapi.json"
    debug: bool = Field(default=False, alias="DEBUG")
    environment: Literal["dev", "prod"] = "dev"

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")  # noqa: S104
    port: int = Field(default=7860, alias="PORT")

    # Database
    database_url: str = Field(
        default="sqlite:///./data/geti_tune.db", alias="DATABASE_URL", description="Database connection URL"
    )
    db_echo: bool = Field(default=False, alias="DB_ECHO")

    # Alembic
    alembic_config_path: str = "app/alembic.ini"
    alembic_script_location: str = "app/alembic"

    # Proxy settings
    no_proxy: str = Field(default="localhost,127.0.0.1,::1", alias="no_proxy")

    @property
    def database_dir(self) -> Path:
        """Get database directory path"""
        if self.database_url.startswith("sqlite:///"):
            db_path = Path(self.database_url.replace("sqlite:///", ""))
            return db_path.parent
        return Path("./data")


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()
