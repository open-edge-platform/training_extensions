# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path
from typing import cast

from behave.runner import Context

from .base import BaseServerRunner


class ProcessRunner(BaseServerRunner):
    def __init__(self, context: Context):
        super().__init__(context)
        self.process = None

    def start_server(self):
        port = 7861
        tmp_dir = cast(Path, self.tmp_dir)
        env = {
            **os.environ.copy(),
            "DATA_DIR": str(tmp_dir / "data"),
            "LOG_DIR": str(tmp_dir / "logs"),
            "PORT": str(port),
        }

        self.process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", str(port)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.base_url = f"http://localhost:{port}"
        self.context.base_url = self.base_url

    def stop_server(self):
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
