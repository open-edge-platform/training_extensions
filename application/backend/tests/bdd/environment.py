# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from typing import cast

import requests
from behave import fixture, use_fixture
from behave.model import Scenario
from behave.runner import Context


@fixture
def fastapi_server(context: Context) -> Generator[None]:
    """Start FastAPI server as subprocess for testing."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="behave_test_"))
    context.tmp_path = tmp_dir

    # Set up test environment
    env = {
        **os.environ.copy(),
        "DATA_DIR": str(tmp_dir / "data"),
        "LOG_DIR": str(tmp_dir / "logs"),
        "PORT": "7861",
    }

    # Start server subprocess
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7861"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start
    context.base_url = "http://localhost:7861"
    max_retries = 30
    for _ in range(max_retries):
        try:
            response = requests.get(f"{context.base_url}/health", timeout=1)
            if response.status_code == 200:
                break
        except requests.ConnectionError:
            time.sleep(0.2)
    else:
        server_process.kill()
        raise RuntimeError("Server failed to start")

    yield

    # Cleanup
    try:
        server_process.terminate()
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_process.kill()

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)


def before_all(context: Context) -> None:
    """Set up the server before each scenario."""
    use_fixture(fastapi_server, context)


def after_scenario(context: Context, _: Scenario) -> None:
    """Delete dataset .zip archive after each scenario."""
    if hasattr(context, "staged_dataset_path"):
        dataset_path = cast(Path, context.staged_dataset_path)
        if dataset_path.exists():
            dataset_path.unlink()
