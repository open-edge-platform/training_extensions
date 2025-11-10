# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from os import getenv

LOG_DIR = getenv("LOG_DIR", "./logs")
TB_LOG_DIR = os.path.join(LOG_DIR, "tensorboard")
LOG_LEVEL = getenv("LOG_LEVEL", "INFO")


@dataclass
class LogConfig:
    """Configuration for logging behavior."""

    log_file: str = "app.log"
    rotation: str = "10 MB"
    retention: str = "10 days"
    level: str = LOG_LEVEL
    serialize: bool = True
    log_folder: str = LOG_DIR
    tensorboard_log_path: str = TB_LOG_DIR
