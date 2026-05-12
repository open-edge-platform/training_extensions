# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import shutil
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path

import requests
from behave.runner import Context


class BaseServerRunner(ABC):
    """Abstract base strategy for starting the FastAPI server."""

    def __init__(self, context: Context):
        self.context = context
        self.tmp_dir: Path | None = None
        self.base_url: str = ""

    def setup(self):
        """Prepare temporary directories and common setup."""
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="behave_test_"))
        self.context.tmp_path = self.tmp_dir
        for subdir in ["data", "logs"]:
            (self.tmp_dir / subdir).mkdir()

    @abstractmethod
    def start_server(self):
        """Start the actual server process/container."""

    @abstractmethod
    def stop_server(self):
        """Stop the server process/container."""

    def wait_for_health(self, max_retries=30):
        """Common logic to wait for health endpoint."""
        for _ in range(max_retries):
            try:
                response = requests.get(f"{self.base_url}/health", timeout=1)
                if response.status_code == 200:
                    return
            except requests.ConnectionError:
                time.sleep(0.2)
        raise RuntimeError("Server failed to become healthy.")

    def cleanup(self):
        """Remove temporary directories."""
        if self.tmp_dir and self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
