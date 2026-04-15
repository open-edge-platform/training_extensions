# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoints."""

from datetime import timedelta
from time import time

from getitune.cli.cli import CLI


def main() -> None:
    """Entry point for Geti Tune CLI.

    This function is a single entry point for all Geti Tune CLI related operations:
    """
    start = time()
    CLI()
    dt = timedelta(seconds=time() - start)
    print(f"Elapsed time: {dt}")


if __name__ == "__main__":
    main()
