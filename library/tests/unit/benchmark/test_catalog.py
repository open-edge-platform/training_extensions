# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for otx.benchmark.catalog."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from otx.benchmark.catalog import (
    DatasetCatalog,
    DatasetEntry,
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
            script: "scripts/benchmark_datasets/prepare_ds_tiny.py"
            size_tier: tiny
          - name: ds_small
            script: "scripts/benchmark_datasets/prepare_ds_small.py"
            size_tier: small
          - name: cls_tiny
            script: "scripts/benchmark_datasets/prepare_cls_tiny.py"
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
        assert entry.script == "scripts/benchmark_datasets/prepare_ds_tiny.py"
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
            script="scripts/prepare.py",
            size_tier="tiny",
        )
        assert entry.relative_path == Path("my_ds")


# ---------------------------------------------------------------------------
# Helper: create a simple preparation script
# ---------------------------------------------------------------------------


def _make_prep_script(script_path: Path, file_content: str = "world") -> None:
    """Create a minimal preparation script that creates the dataset directory."""
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(
        textwrap.dedent(f"""\
        import argparse
        from pathlib import Path

        parser = argparse.ArgumentParser()
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--name", type=str, required=True)
        args = parser.parse_args()

        dest = args.output_dir / args.name
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "hello.txt").write_text("{file_content}")
    """)
    )


class TestProvisionDataset:
    def test_skips_when_directory_exists(self, tmp_path: Path) -> None:
        """If the dataset directory already exists, the script should not run."""
        data_root = tmp_path / "data"
        ds_dir = data_root / "cached_ds"
        ds_dir.mkdir(parents=True)

        entry = DatasetEntry(
            name="cached_ds",
            script="scripts/prepare.py",
            size_tier="tiny",
        )
        # No need to mock _resolve_script_path — it should never be called
        result = provision_dataset(entry, data_root)
        assert result == ds_dir

    def test_run_script_and_provision(self, tmp_path: Path) -> None:
        """Script should be run and dataset dir created."""
        data_root = tmp_path / "data"

        script_path = tmp_path / "scripts" / "prepare.py"
        _make_prep_script(script_path)

        entry = DatasetEntry(
            name="test_ds",
            script="scripts/prepare.py",
            size_tier="tiny",
        )

        with patch("otx.benchmark.catalog._resolve_script_path", return_value=script_path):
            result = provision_dataset(entry, data_root)

        assert result.exists()
        assert (result / "hello.txt").exists()
        assert (result / "hello.txt").read_text() == "world"

    def test_script_not_found_raises(self, tmp_path: Path) -> None:
        """Missing preparation script must raise FileNotFoundError."""
        data_root = tmp_path / "data"
        entry = DatasetEntry(
            name="missing",
            script="scripts/does_not_exist.py",
            size_tier="tiny",
        )

        with patch(
            "otx.benchmark.catalog._resolve_script_path",
            return_value=tmp_path / "scripts" / "does_not_exist.py",
        ), pytest.raises(FileNotFoundError, match="Preparation script not found"):
            provision_dataset(entry, data_root)

    def test_script_failure_raises(self, tmp_path: Path) -> None:
        """A script that exits non-zero must raise RuntimeError."""
        data_root = tmp_path / "data"
        script_path = tmp_path / "scripts" / "bad.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("import sys; sys.exit(1)\n")

        entry = DatasetEntry(
            name="bad_ds",
            script="scripts/bad.py",
            size_tier="tiny",
        )

        with patch("otx.benchmark.catalog._resolve_script_path", return_value=script_path), pytest.raises(
            RuntimeError, match="failed with exit code"
        ):
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
                script: "scripts/prepare.py"
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
                script: "scripts/prepare.py"
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
# Provision multiple datasets
# ---------------------------------------------------------------------------


class TestProvisionDatasets:
    def test_provisions_all_entries(self, tmp_path: Path) -> None:
        script_a = tmp_path / "scripts" / "prepare_a.py"
        script_b = tmp_path / "scripts" / "prepare_b.py"
        _make_prep_script(script_a, file_content="content_a")
        _make_prep_script(script_b, file_content="content_b")

        entry_a = DatasetEntry(name="a", script="scripts/prepare_a.py", size_tier="tiny")
        entry_b = DatasetEntry(name="b", script="scripts/prepare_b.py", size_tier="tiny")
        catalog = DatasetCatalog(version=1, datasets={"a": entry_a, "b": entry_b})

        data_root = tmp_path / "data"

        def resolve(script: str) -> Path:
            return tmp_path / script

        with patch("otx.benchmark.catalog._resolve_script_path", side_effect=resolve):
            result = provision_datasets(catalog, data_root)

        assert set(result.keys()) == {"a", "b"}
        assert (result["a"] / "hello.txt").exists()
        assert (result["b"] / "hello.txt").exists()

    def test_provisions_subset(self, tmp_path: Path) -> None:
        script_a = tmp_path / "scripts" / "prepare_a.py"
        _make_prep_script(script_a, file_content="data")

        entry_a = DatasetEntry(name="a", script="scripts/prepare_a.py", size_tier="tiny")
        entry_b = DatasetEntry(name="b", script="scripts/prepare_b.py", size_tier="tiny")
        catalog = DatasetCatalog(version=1, datasets={"a": entry_a, "b": entry_b})

        data_root = tmp_path / "data"

        def resolve(script: str) -> Path:
            return tmp_path / script

        with patch("otx.benchmark.catalog._resolve_script_path", side_effect=resolve):
            result = provision_datasets(catalog, data_root, entries=[entry_a])

        assert "a" in result
        assert "b" not in result
