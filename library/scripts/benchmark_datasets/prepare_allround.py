#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare the 'allround' benchmark dataset."""

from __future__ import annotations

from getitune.benchmark.dataset_helpers import download, extract_archive, parse_args

_URL = "https://storage.geti.intel.com/test-data/integration-iai/datasets/allround.zip"


def main() -> None:
    """Download and prepare the allround benchmark dataset."""
    args = parse_args(description="Prepare the allround benchmark dataset.")

    archive = download(_URL, dest_dir=args.archive_dir, filename=f"{args.name}.zip")
    extract_archive(archive, args.dest)
    archive.unlink(missing_ok=True)

    print(f"Dataset '{args.name}' ready at {args.dest}")


if __name__ == "__main__":
    main()
