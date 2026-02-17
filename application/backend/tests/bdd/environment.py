# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import shutil
import tempfile
import time
from collections.abc import Generator
from pathlib import Path

import requests
from behave import fixture, use_fixture
from behave.runner import Context
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs


@fixture
def fastapi_server(context: Context) -> Generator[None]:
    """Start FastAPI server in a Docker container for testing."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="behave_test_"))
    context.tmp_path = tmp_dir
    data_dir = tmp_dir / "data"
    logs_dir = tmp_dir / "logs"
    data_dir.mkdir()
    logs_dir.mkdir()

    image_tag = os.getenv("IMAGE_TAG")
    if image_tag is None:
        raise RuntimeError("Environment variable IMAGE_TAG not set")

    port = 7860
    container = (
        DockerContainer(image=image_tag)
        .with_env("DATA_DIR", "/application/backend/data")
        .with_env("LOG_DIR", "/application/backend/logs")
        .with_volume_mapping(str(data_dir), "/application/backend/data", mode="rw")
        .with_volume_mapping(str(logs_dir), "/application/backend/logs", mode="rw")
        .with_bind_ports(port, port)
    )

    with container as server:
        # Wait for the server to be ready
        wait_for_logs(server, "Application startup completed", timeout=60)

        host = server.get_container_host_ip()
        context.base_url = f"http://{host}:{port}"

        # Wait for the application to be healthy
        max_retries = 30
        for _ in range(max_retries):
            try:
                response = requests.get(f"{context.base_url}/health", timeout=1)
                if response.status_code == 200:
                    break
            except requests.ConnectionError:
                time.sleep(0.2)
        else:
            raise RuntimeError("Server failed to start or become healthy.")

        yield

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)


def before_all(context: Context) -> None:
    """Set up the server before each scenario."""
    use_fixture(fastapi_server, context)
