# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for otx.benchmark.dataset_helpers."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from otx.benchmark.dataset_helpers import (
    DatasetArgs,
    download,
    extract_archive,
    parse_args,
)

# ---------------------------------------------------------------------------
# DatasetArgs
# ---------------------------------------------------------------------------


class TestDatasetArgs:
    def test_dest_property(self) -> None:
        args = DatasetArgs(output_dir=Path("/data"), name="my_ds")
        assert args.dest == Path("/data/my_ds")

    def test_archive_dir_property(self) -> None:
        args = DatasetArgs(output_dir=Path("/data"), name="my_ds")
        assert args.archive_dir == Path("/data/.archives")


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_parses_required_args(self, tmp_path: Path) -> None:
        data_dir = str(tmp_path / "data")
        with patch("sys.argv", ["prog", "--output-dir", data_dir, "--name", "ds_a"]):
            args = parse_args()
        assert args.output_dir == Path(data_dir)
        assert args.name == "ds_a"
        assert args.dest == Path(data_dir) / "ds_a"

    def test_missing_args_exits(self) -> None:
        with patch("sys.argv", ["prog"]), pytest.raises(SystemExit):
            parse_args()


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


class TestDownload:
    def test_downloads_to_dest(self, tmp_path: Path) -> None:
        dest_dir = tmp_path / "archives"

        # Mock urlretrieve to just create a file
        def fake_urlretrieve(url: str, dest: str | Path) -> None:
            Path(dest).write_text("fake archive content")

        with patch("otx.benchmark.dataset_helpers.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            result = download("https://example.com/data.tar.gz", dest_dir)

        assert result == dest_dir / "data.tar.gz"
        assert result.read_text() == "fake archive content"

    def test_custom_filename(self, tmp_path: Path) -> None:
        dest_dir = tmp_path / "archives"

        def fake_urlretrieve(url: str, dest: str | Path) -> None:
            Path(dest).write_text("content")

        with patch("otx.benchmark.dataset_helpers.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            result = download("https://example.com/v2/data.tar.gz", dest_dir, filename="custom.tar.gz")

        assert result == dest_dir / "custom.tar.gz"

    def test_creates_dest_dir(self, tmp_path: Path) -> None:
        dest_dir = tmp_path / "nested" / "deep" / "dir"
        assert not dest_dir.exists()

        def fake_urlretrieve(url: str, dest: str | Path) -> None:
            Path(dest).write_text("content")

        with patch("otx.benchmark.dataset_helpers.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            download("https://example.com/a.zip", dest_dir)

        assert dest_dir.exists()


# ---------------------------------------------------------------------------
# extract_archive
# ---------------------------------------------------------------------------


def _make_tar_gz(path: Path, files: dict[str, str]) -> None:
    """Create a .tar.gz archive with the given files."""
    with tarfile.open(path, "w:gz") as tf:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def _make_zip(path: Path, files: dict[str, str]) -> None:
    """Create a .zip archive with the given files."""
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)


class TestExtractArchive:
    def test_extract_tar_gz(self, tmp_path: Path) -> None:
        archive = tmp_path / "data.tar.gz"
        _make_tar_gz(archive, {"file.txt": "hello"})

        dest = tmp_path / "output"
        result = extract_archive(archive, dest)

        assert result == dest
        assert (dest / "file.txt").read_text() == "hello"

    def test_extract_zip(self, tmp_path: Path) -> None:
        archive = tmp_path / "data.zip"
        _make_zip(archive, {"readme.md": "# Readme"})

        dest = tmp_path / "output"
        result = extract_archive(archive, dest)

        assert result == dest
        assert (dest / "readme.md").read_text() == "# Readme"

    def test_clean_dest_removes_existing(self, tmp_path: Path) -> None:
        dest = tmp_path / "output"
        dest.mkdir()
        (dest / "stale.txt").write_text("old content")

        archive = tmp_path / "data.zip"
        _make_zip(archive, {"fresh.txt": "new"})

        extract_archive(archive, dest, clean_dest=True)

        assert (dest / "fresh.txt").exists()
        assert not (dest / "stale.txt").exists()

    def test_no_clean_dest_preserves_existing(self, tmp_path: Path) -> None:
        dest = tmp_path / "output"
        dest.mkdir()
        (dest / "keep.txt").write_text("kept")

        archive = tmp_path / "data.zip"
        _make_zip(archive, {"added.txt": "new"})

        extract_archive(archive, dest, clean_dest=False)

        assert (dest / "added.txt").exists()
        assert (dest / "keep.txt").exists()

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.dat"
        bad_file.write_text("not an archive")
        with pytest.raises(ValueError, match="Unsupported archive format"):
            extract_archive(bad_file, tmp_path / "out")
