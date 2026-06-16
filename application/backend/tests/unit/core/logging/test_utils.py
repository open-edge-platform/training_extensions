# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import sys

from app.core.logging import LogConfig, logging_ctx


def _read_messages(log_path) -> list[str]:
    return [json.loads(line)["record"]["message"] for line in log_path.read_text().splitlines()]


class TestLoggingCtx:
    def test_logging_ctx_captures_stdlib_and_stream_output(self, tmp_path):
        log_path = tmp_path / "job.log"

        with logging_ctx(LogConfig(log_folder=str(tmp_path), log_file=log_path.name, level="INFO", serialize=True)):
            logging.getLogger("nncf").info("Metric of initial model: 0.5090429186820984")
            print("Collecting values for each data item using the initial model")
            print("Elapsed Time: 00:00:53", file=sys.stderr)

        messages = _read_messages(log_path)
        assert "Metric of initial model: 0.5090429186820984" in messages
        assert "Collecting values for each data item using the initial model" in messages
        assert "Elapsed Time: 00:00:53" in messages

    def test_logging_ctx_keeps_only_latest_carriage_return_progress_update(self, tmp_path):
        log_path = tmp_path / "job.log"

        with logging_ctx(LogConfig(log_folder=str(tmp_path), log_file=log_path.name, level="INFO", serialize=True)):
            sys.stderr.write("Collecting metrics 50%\r")
            sys.stderr.write("Collecting metrics 100%\n")

        messages = _read_messages(log_path)

        assert "Collecting metrics 50%" not in messages
        assert "Collecting metrics 100%" in messages
