# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Application configuration management"""

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
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
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    environment: Literal["dev", "prod"] = "dev"
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    log_dir: Path = Field(default=Path("logs"), alias="LOG_DIR")
    worker_dir: Path | None = None
    job_dir: Path | None = None

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")  # noqa: S104
    port: int = Field(default=7860, alias="PORT")
    static_files_dir: Path | None = Field(
        default=None,
        alias="STATIC_FILES_DIR",
        description="Directory containing static UI files",
    )

    # CORS
    cors_origins: str = Field(
        default="http://localhost:3000, http://localhost:7860",
        alias="CORS_ORIGINS",
    )

    # Database
    database_file: str = Field(default="geti_tune.db", alias="DATABASE_FILE", description="Database filename")
    db_echo: bool = Field(default=False, alias="DB_ECHO")

    # Alembic
    alembic_config_path: str = "app/alembic.ini"
    alembic_script_location: str = "app/alembic"

    # Proxy settings
    no_proxy: str = Field(default="localhost,127.0.0.1,::1", alias="no_proxy")

    gpu_slots: int = Field(default=1, alias="GPU_SLOTS", description="Number of GPU slots available for model tuning")

    @property
    def database_url(self) -> str:
        """Get database URL"""
        return f"sqlite:///{self.data_dir / self.database_file}"

    @property
    def cors_allowed_origins(self) -> list[str]:
        """Parsed list of allowed CORS origins."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @model_validator(mode="after")
    def set_default_dirs(self) -> "Settings":
        """Set default directories based on log_dir"""
        if self.worker_dir is None:
            self.worker_dir = self.log_dir / "workers"
        if self.job_dir is None:
            self.job_dir = self.log_dir / "jobs"

        return self

    def ensure_dirs_exist(self) -> None:
        """Create all directories if they don't exist."""
        for d in [self.data_dir, self.log_dir, self.worker_dir, self.job_dir]:
            if d:
                d.mkdir(parents=True, exist_ok=True)

    @field_validator("static_files_dir", "alembic_config_path", "alembic_script_location", mode="after")
    def prefix_paths(cls, v: str | Path | None) -> str | Path | None:
        # In "frozen" pyinstaller applications data paths must be prefixed with the absolute path to the bundle folder
        # which is stored in  sys._MEIPASS attribute.
        # https://pyinstaller.org/en/stable/runtime-information.html
        if v and getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            # If application is running in pyinstaller bundle, adjust the path accordingly.
            prefixed_path = os.path.join(getattr(sys, "_MEIPASS", ""), v)
            return Path(prefixed_path) if isinstance(v, Path) else prefixed_path
        return v


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()
