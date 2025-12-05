# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import subprocess
import sys
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

from otx.cli import main

ExportCase2Test = namedtuple("ExportCase2Test", ["export_format", "export_demo_package", "expected_output"])


def run_main(command_cfg: list[str], open_subprocess: bool) -> None:
    if open_subprocess:
        _run_main_with_open_subprocess(command_cfg)
    else:
        _run_main(command_cfg)


def get_tests_asset_path(*relative_parts: str) -> str:
    """Return absolute path to an item under tests/assets regardless of CWD.

    Usage:
        get_tests_asset_path("car_tree_bug")
        get_tests_asset_path("geti", "model_configs", "detection.yaml")
    """
    base = Path(__file__).resolve().parent / "assets"
    return str(base.joinpath(*relative_parts))


def _run_main_with_open_subprocess(command_cfg) -> None:
    try:
        subprocess.run(
            [sys.executable, __file__, *command_cfg],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode()
        msg = f"Fail to run main: stderr={stderr}"
        logging.exception(msg)
        raise


def _run_main(command_cfg) -> None:
    with patch("sys.argv", command_cfg):
        main()


if __name__ == "__main__":
    _run_main(sys.argv[1:])
