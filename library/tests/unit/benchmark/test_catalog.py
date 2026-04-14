# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for otx.benchmark.catalog."""

from __future__ import annotations

import tarfile
import textwrap
from pathlib import Path

import pytest

from otx.benchmark.catalog import (
    DatasetCatalog,
    DatasetEntry,
    _extract,
    _read_sentinel,
    _sha256_of_file,
    _write_sentinel,
    load_catalog,
    provision_dataset,
    provision_datasets,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def catalog_yaml(tmp_path: Path) -> Path:
    """Write a minimal catalog YAML and return its path."""
    content = textwrap.dedent("""\
        version: 1
        datasets:
          - name: ds_tiny
            url: "https://example.com/ds_tiny.tar.gz"
            sha256: "aaa"
            size_tier: tiny
          - name: ds_small
            url: "https://example.com/ds_small.tar.gz"
            sha256: "bbb"
            size_tier: small
          - name: cls_tiny
            url: "https://example.com/cls_tiny.tar.gz"
            sha256: "ccc"
            size_tier: tiny
    """)
    p = tmp_path / "catalog.yaml"
    p.write_text(content)
    return p


@pytest.fixture
def catalog(catalog_yaml: Path) -> DatasetCatalog:
    return load_catalog(catalog_yaml)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class TestLoadCatalog:
    def test_loads_version(self, catalog: DatasetCatalog) -> None:
        assert catalog.version == 1

    def test_parses_all_entries(self, catalog: DatasetCatalog) -> None:
        assert len(catalog.all_entries()) == 3

    def test_dataset_keys(self, catalog: DatasetCatalog) -> None:
        assert set(catalog.datasets.keys()) == {"ds_tiny", "ds_small", "cls_tiny"}

    def test_entry_fields(self, catalog: DatasetCatalog) -> None:
        entry = catalog.get("ds_tiny")
        assert entry.name == "ds_tiny"
        assert entry.sha256 == "aaa"
        assert entry.size_tier == "tiny"

    def test_get_unknown_raises(self, catalog: DatasetCatalog) -> None:
        with pytest.raises(KeyError, match="not_real"):
            catalog.get("not_real")


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestCatalogFilter:
    def test_filter_by_size_tier(self, catalog: DatasetCatalog) -> None:
        results = catalog.filter(size_tiers=["tiny"])
        assert len(results) == 2
        names = {e.name for e in results}
        assert names == {"ds_tiny", "cls_tiny"}

    def test_filter_by_name(self, catalog: DatasetCatalog) -> None:
        results = catalog.filter(names={"ds_small"})
        assert len(results) == 1
        assert results[0].name == "ds_small"

    def test_combined_filters(self, catalog: DatasetCatalog) -> None:
        results = catalog.filter(size_tiers=["tiny"], names={"ds_tiny"})
        assert len(results) == 1
        assert results[0].name == "ds_tiny"

    def test_no_match_returns_empty(self, catalog: DatasetCatalog) -> None:
        assert catalog.filter(size_tiers=["large"]) == []


# ---------------------------------------------------------------------------
# Relative path
# ---------------------------------------------------------------------------


class TestDatasetEntry:
    def test_relative_path(self) -> None:
        entry = DatasetEntry(
            name="my_ds",
            url="",
            sha256="",
            size_tier="tiny",
        )
        assert entry.relative_path == Path("my_ds")


# ---------------------------------------------------------------------------
# Provisioning (with a real tiny archive)
# ---------------------------------------------------------------------------


def _make_tar_gz(archive_path: Path, inner_files: dict[str, str]) -> str:
    """Create a tar.gz with the given files and return its sha256."""
    with tarfile.open(archive_path, "w:gz") as tf:
        for name, content in inner_files.items():
            import io

            data = content.encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return _sha256_of_file(archive_path)


class TestProvisionDataset:
    def test_cache_hit(self, tmp_path: Path) -> None:
        """If the sentinel matches, no download should happen."""
        data_root = tmp_path / "data"
        ds_dir = data_root / "cached_ds"
        ds_dir.mkdir(parents=True)
        _write_sentinel(ds_dir, "match_me")

        entry = DatasetEntry(
            name="cached_ds",
            url="https://will-not-be-called.example.com/x.tar.gz",
            sha256="match_me",
            size_tier="tiny",
        )
        # Should NOT attempt any download (url is unreachable)
        result = provision_dataset(entry, data_root)
        assert result == ds_dir

    def test_download_and_extract(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Simulate a successful download + extraction cycle."""
        data_root = tmp_path / "data"
        archive_dir = tmp_path / "serve"
        archive_dir.mkdir()
        archive_path = archive_dir / "test_ds.tar.gz"
        real_sha = _make_tar_gz(archive_path, {"hello.txt": "world"})

        # Monkeypatch _download to copy the local file instead of HTTP
        def fake_download(url: str, dest: Path) -> None:
            import shutil

            shutil.copy(archive_path, dest)

        monkeypatch.setattr("otx.benchmark.catalog._download", fake_download)

        entry = DatasetEntry(
            name="test_ds",
            url="https://fake.example.com/test_ds.tar.gz",
            sha256=real_sha,
            size_tier="tiny",
        )
        result = provision_dataset(entry, data_root)
        assert result.exists()
        assert (result / "hello.txt").exists()
        # Sentinel should be written
        assert (result / ".sha256").read_text().strip() == real_sha

    def test_checksum_mismatch_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A bad checksum must raise RuntimeError."""
        data_root = tmp_path / "data"
        archive_dir = tmp_path / "serve"
        archive_dir.mkdir()
        archive_path = archive_dir / "bad.tar.gz"
        _make_tar_gz(archive_path, {"x.txt": "y"})

        def fake_download(url: str, dest: Path) -> None:
            import shutil

            shutil.copy(archive_path, dest)

        monkeypatch.setattr("otx.benchmark.catalog._download", fake_download)

        entry = DatasetEntry(
            name="bad_ds",
            url="https://fake.example.com/bad.tar.gz",
            sha256="wrong_sha",
            size_tier="tiny",
        )
        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            provision_dataset(entry, data_root)


# ---------------------------------------------------------------------------
# Extra fields & edge cases
# ---------------------------------------------------------------------------


class TestLoadCatalogExtras:
    def test_unknown_keys_go_to_extra(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            version: 1
            datasets:
              - name: ds_custom
                url: "https://example.com/ds.tar.gz"
                sha256: "abc"
                size_tier: tiny
                custom_field: 42
                another_field: hello
        """)
        p = tmp_path / "catalog.yaml"
        p.write_text(content)
        catalog = load_catalog(p)
        entry = catalog.get("ds_custom")
        assert entry.extra["custom_field"] == 42
        assert entry.extra["another_field"] == "hello"

    def test_default_version(self, tmp_path: Path) -> None:
        """Catalog without explicit version should default to 1."""
        content = textwrap.dedent("""\
            datasets:
              - name: ds
                url: "https://example.com/ds.tar.gz"
                sha256: "x"
                size_tier: tiny
        """)
        p = tmp_path / "catalog.yaml"
        p.write_text(content)
        catalog = load_catalog(p)
        assert catalog.version == 1

    def test_empty_datasets_section(self, tmp_path: Path) -> None:
        content = "version: 1\ndatasets: []\n"
        p = tmp_path / "catalog.yaml"
        p.write_text(content)
        catalog = load_catalog(p)
        assert catalog.all_entries() == []

    def test_filter_no_args_returns_all(self, catalog: DatasetCatalog) -> None:
        """Calling filter with no arguments returns everything."""
        results = catalog.filter()
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Sentinel helpers
# ---------------------------------------------------------------------------


class TestSentinel:
    def test_write_and_read_sentinel(self, tmp_path: Path) -> None:
        ds_dir = tmp_path / "ds"
        ds_dir.mkdir()
        _write_sentinel(ds_dir, "abc123")
        assert _read_sentinel(ds_dir) == "abc123"

    def test_read_sentinel_missing(self, tmp_path: Path) -> None:
        assert _read_sentinel(tmp_path / "nope") is None


# ---------------------------------------------------------------------------
# Archive extraction
# ---------------------------------------------------------------------------


class TestExtract:
    def test_extract_zip(self, tmp_path: Path) -> None:
        import zipfile

        archive_path = tmp_path / "test.zip"
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr("hello.txt", "world")

        dest = tmp_path / "extracted"
        _extract(archive_path, dest)
        assert (dest / "hello.txt").exists()
        assert (dest / "hello.txt").read_text() == "world"

    def test_extract_tar_gz(self, tmp_path: Path) -> None:
        archive_path = tmp_path / "test.tar.gz"
        _make_tar_gz(archive_path, {"data.txt": "content"})

        dest = tmp_path / "extracted"
        _extract(archive_path, dest)
        assert (dest / "data.txt").exists()

    def test_extract_unsupported_format_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.dat"
        bad_file.write_text("not an archive")
        with pytest.raises(ValueError, match="Unsupported archive format"):
            _extract(bad_file, tmp_path / "out")


class TestProvisionDatasetAdvanced:
    def test_redownload_on_sha_mismatch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """If sentinel has a different sha, dataset should be re-downloaded."""
        data_root = tmp_path / "data"
        ds_dir = data_root / "evolving_ds"
        ds_dir.mkdir(parents=True)
        _write_sentinel(ds_dir, "old_sha")

        archive_dir = tmp_path / "serve"
        archive_dir.mkdir()
        archive_path = archive_dir / "evolving_ds.tar.gz"
        new_sha = _make_tar_gz(archive_path, {"updated.txt": "new content"})

        def fake_download(url: str, dest: Path) -> None:
            import shutil

            shutil.copy(archive_path, dest)

        monkeypatch.setattr("otx.benchmark.catalog._download", fake_download)

        entry = DatasetEntry(
            name="evolving_ds",
            url="https://example.com/evolving_ds.tar.gz",
            sha256=new_sha,
            size_tier="tiny",
        )
        result = provision_dataset(entry, data_root)
        assert result.exists()
        assert (result / "updated.txt").exists()
        assert _read_sentinel(result) == new_sha

    def test_download_and_extract_zip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Provision should work with .zip archives too."""
        import zipfile

        data_root = tmp_path / "data"
        archive_dir = tmp_path / "serve"
        archive_dir.mkdir()
        archive_path = archive_dir / "zip_ds.zip"
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr("file.txt", "zip content")
        real_sha = _sha256_of_file(archive_path)

        def fake_download(url: str, dest: Path) -> None:
            import shutil

            shutil.copy(archive_path, dest)

        monkeypatch.setattr("otx.benchmark.catalog._download", fake_download)

        entry = DatasetEntry(
            name="zip_ds",
            url="https://example.com/zip_ds.zip",
            sha256=real_sha,
            size_tier="tiny",
        )
        result = provision_dataset(entry, data_root)
        assert (result / "file.txt").read_text() == "zip content"


class TestProvisionDatasets:
    def test_provisions_all_entries(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        archive_dir = tmp_path / "serve"
        archive_dir.mkdir()
        archive_a = archive_dir / "a.tar.gz"
        archive_b = archive_dir / "b.tar.gz"
        sha_a = _make_tar_gz(archive_a, {"a.txt": "content_a"})
        sha_b = _make_tar_gz(archive_b, {"b.txt": "content_b"})

        def fake_download(url: str, dest: Path) -> None:
            import shutil

            name = Path(url).name
            shutil.copy(archive_dir / name, dest)

        monkeypatch.setattr("otx.benchmark.catalog._download", fake_download)

        entry_a = DatasetEntry(name="a", url="https://x.com/a.tar.gz", sha256=sha_a, size_tier="tiny")
        entry_b = DatasetEntry(name="b", url="https://x.com/b.tar.gz", sha256=sha_b, size_tier="tiny")
        catalog = DatasetCatalog(version=1, datasets={"a": entry_a, "b": entry_b})

        data_root = tmp_path / "data"
        result = provision_datasets(catalog, data_root)
        assert set(result.keys()) == {"a", "b"}
        assert (result["a"] / "a.txt").exists()
        assert (result["b"] / "b.txt").exists()

    def test_provisions_subset(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        archive_dir = tmp_path / "serve"
        archive_dir.mkdir()
        archive = archive_dir / "only.tar.gz"
        sha = _make_tar_gz(archive, {"only.txt": "data"})

        def fake_download(url: str, dest: Path) -> None:
            import shutil

            shutil.copy(archive, dest)

        monkeypatch.setattr("otx.benchmark.catalog._download", fake_download)

        entry_a = DatasetEntry(name="a", url="https://x.com/only.tar.gz", sha256=sha, size_tier="tiny")
        entry_b = DatasetEntry(name="b", url="https://unreachable.com/b.tar.gz", sha256="nope", size_tier="tiny")
        catalog = DatasetCatalog(version=1, datasets={"a": entry_a, "b": entry_b})

        data_root = tmp_path / "data"
        result = provision_datasets(catalog, data_root, entries=[entry_a])
        assert "a" in result
        assert "b" not in result
