# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import urllib.request
import zipfile
from pathlib import Path

import pytest

BUCKET_NAME = "geti-datasets"
OBJECT_NAME = "regression.zip"
PARENT_DIR = Path(__file__).parent
DATASETS_DIR = PARENT_DIR / "regression"


def _download_regression_datasets(dest_dir: Path) -> None:
    """Download and unpack the regression dataset archive from GCS (public bucket)."""
    archive = dest_dir / OBJECT_NAME

    if not archive.exists():
        url = f"https://storage.googleapis.com/{BUCKET_NAME}/{OBJECT_NAME}"
        urllib.request.urlretrieve(url, archive)

    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest_dir)


def pytest_configure() -> None:
    """Session-wide hook - download datasets before collection begins."""
    _download_regression_datasets(PARENT_DIR)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize after datasets are already downloaded."""
    if "archive" in metafunc.fixturenames:
        zip_files = sorted(DATASETS_DIR.glob("*.zip"))
        if not zip_files:
            raise pytest.UsageError(
                f"No regression dataset archives were found in '{DATASETS_DIR}'. The dataset download/extraction may "
                f"have failed, or the archive structure may have changed."
            )
        metafunc.parametrize("archive", zip_files, ids=[p.stem for p in zip_files])
