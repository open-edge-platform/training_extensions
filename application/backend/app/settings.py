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
    app_name: str = "Geti"
    version: str = "0.1.0"
    summary: str = "Geti server"
    description: str = (
        "Geti allows to fine-tune computer vision models at the edge. "
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
    staged_datasets_dir: Path | None = None

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
    database_file: str = Field(default="geti.db", alias="DATABASE_FILE", description="Database filename")
    db_echo: bool = Field(default=False, alias="DB_ECHO")

    # Alembic
    alembic_config_path: str = "app/alembic.ini"
    alembic_script_location: str = "app/alembic"

    # Proxy settings
    no_proxy: str = Field(default="localhost,127.0.0.1,::1", alias="no_proxy")

    gpu_slots: int = Field(default=1, alias="GPU_SLOTS", description="Number of GPU slots available for model tuning")

    # WebRTC
    webrtc_advertise_ip: str | None = Field(default=None, alias="WEBRTC_ADVERTISE_IP")

    # Simplified WebRTC config
    coturn_host: str | None = Field(default=None, alias="COTURN_HOST")
    coturn_port: int = Field(default=443, alias="COTURN_PORT")
    coturn_username: str = Field(default="user", alias="COTURN_USERNAME")
    coturn_password: str = Field(default="password", alias="COTURN_PASSWORD")
    stun_server: str | None = Field(default=None, alias="STUN_SERVER")

    # Inference
    inference_media_limit: int = Field(
        default=10,
        alias="INFERENCE_MEDIA_LIMIT",
        description="Maximum number for images or video frames passed for inference",
    )
    inference_model_ttl: int = Field(
        default=60,
        alias="INFERENCE_MODEL_TTL",
        description="Time to live for a model loaded for inference, before unloading",
    )

    # Video
    video_cache_enabled: bool = Field(
        default=True,
        alias="GETI_VIDEO_CACHE_ENABLED",
        description="Whether to enable video frame caching for faster repeated access",
    )
    video_cache_ttl: float = Field(
        default=30.0,
        alias="GETI_VIDEO_CACHE_TTL",
        description="Time-to-live in seconds for cached video handles before eviction",
    )
    video_cache_cleanup_interval: float = Field(
        default=5.0,
        alias="GETI_VIDEO_CACHE_CLEANUP_INTERVAL",
        description="Interval in seconds between cache cleanup sweeps",
    )
    video_cache_max_frames_per_video: int = Field(
        default=100,
        alias="GETI_VIDEO_CACHE_MAX_FRAMES_PER_VIDEO",
        description="Maximum number of decoded frames cached per video",
    )

    @property
    def ice_servers(self) -> list[dict]:
        """Compute ICE servers from coturn and STUN configuration."""
        servers = []
        if self.coturn_host:
            servers.append(
                {
                    "urls": f"turn:{self.coturn_host}:{self.coturn_port}?transport=tcp",
                    "username": self.coturn_username,
                    "credential": self.coturn_password,
                }
            )

        if self.stun_server:
            servers.append({"urls": self.stun_server})

        return servers

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
        if self.staged_datasets_dir is None:
            self.staged_datasets_dir = self.data_dir / "staged_datasets"

        return self

    def ensure_dirs_exist(self) -> None:
        """Create all directories if they don't exist."""
        for d in [self.data_dir, self.log_dir, self.worker_dir, self.job_dir, self.staged_datasets_dir]:
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

    @field_validator("stun_server")
    def validate_stun_server(cls, v: str | None) -> str | None:
        """Ensure that STUN server, if set, is a full ICE URL starting with stun: or stuns:."""
        if v:
            if not isinstance(v, str):
                raise TypeError("stun_server must be a string.")
            if not (v.startswith(("stun:", "stuns:"))):
                raise ValueError("stun_server must be a full ICE URL starting with 'stun:' or 'stuns:'.")
        return v


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()
