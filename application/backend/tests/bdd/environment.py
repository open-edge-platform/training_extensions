# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
from collections.abc import Generator

from behave import fixture, use_fixture
from behave.runner import Context

from tests.bdd.server_runner import DockerRunner, ProcessRunner


@fixture
def fastapi_server(context: Context) -> Generator[None]:
    """Fixture that uses the selected strategy."""
    runner_type = os.getenv("RUNNER", "process").lower()
    if runner_type not in ("docker", "process"):
        raise RuntimeError("Environment variable RUNNER must be either unset or set to 'docker' or 'process'")

    runner = DockerRunner(context) if runner_type == "docker" else ProcessRunner(context)
    runner.setup()
    try:
        runner.start_server()
        runner.wait_for_health()
        yield
    finally:
        try:
            runner.stop_server()
        finally:
            if os.getenv("KEEP_ARTIFACTS", "0").lower() not in ("1", "true"):
                runner.cleanup()


def before_all(context: Context) -> None:
    """Set up the server before each scenario."""
    use_fixture(fastapi_server, context)
