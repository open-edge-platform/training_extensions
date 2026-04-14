# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for otx.benchmark.cli."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from otx.benchmark.cli import _build_parser, _cmd_provision, _cmd_run, _parse_key_value_pairs, main

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CATALOG_YAML = textwrap.dedent("""\
    version: 1
    datasets:
      - name: ds_a
        url: "https://example.com/a.tar.gz"
        sha256: "aaa"
        size_tier: tiny
""")

MANIFEST_YAML = textwrap.dedent("""\
    version: 1
    defaults:
      num_seeds: 1
      eval_upto: train
      deterministic: true

    experiments:
      detection:
        models:
          - name: model_a
            priority: core
            recipe: detection/yolox_s.yaml
        datasets:
          - ds_a
        criteria:
          accuracy_metric: mAP
          thresholds:
            "training:val/{metric}": { compare: ">=", margin: 0.10 }
""")


@pytest.fixture
def catalog_file(tmp_path: Path) -> Path:
    p = tmp_path / "catalog.yaml"
    p.write_text(CATALOG_YAML)
    return p


@pytest.fixture
def manifest_file(tmp_path: Path) -> Path:
    p = tmp_path / "manifest.yaml"
    p.write_text(MANIFEST_YAML)
    return p


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_provision_subcommand(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["provision", "--catalog", "cat.yaml", "--data-root", "d/"])
        assert args.command == "provision"
        assert args.catalog == Path("cat.yaml")
        assert args.data_root == Path("d/")

    def test_run_subcommand_defaults(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run"])
        assert args.command == "run"
        assert args.catalog == Path("benchmark_catalog.yaml")
        assert args.manifest == Path("benchmark_manifest.yaml")
        assert args.output_root == Path("results")
        assert args.accelerator == "gpu"
        assert args.deterministic is True
        assert args.dry_run is False

    def test_run_subcommand_filters(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "run",
                "--task",
                "detection",
                "--model",
                "yolox_s",
                "--dataset",
                "ds_a",
                "--priority",
                "core",
                "--size-tier",
                "tiny",
                "--scenario",
                "default",
                "--scenario-tag",
                "configurable",
                "--num-seeds",
                "5",
                "--max-epochs",
                "10",
                "--eval-upto",
                "export",
                "--dry-run",
            ]
        )
        assert args.task == ["detection"]
        assert args.model == ["yolox_s"]
        assert args.dataset == ["ds_a"]
        assert args.priority == ["core"]
        assert args.size_tier == ["tiny"]
        assert args.scenario == ["default"]
        assert args.scenario_tag == ["configurable"]
        assert args.num_seeds == 5
        assert args.max_epochs == 10
        assert args.eval_upto == "export"
        assert args.dry_run is True

    def test_run_no_deterministic(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "--no-deterministic"])
        assert args.deterministic is False

    def test_verbose_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["provision", "-v"])
        assert args.verbose is True

    def test_no_subcommand_errors(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


# ---------------------------------------------------------------------------
# Provision sub-command
# ---------------------------------------------------------------------------


class TestCmdProvision:
    def test_provision_calls_provision_datasets(
        self,
        catalog_file: Path,
        tmp_path: Path,
    ) -> None:
        from otx.benchmark.catalog import load_catalog

        load_catalog(catalog_file)

        parser = _build_parser()
        args = parser.parse_args(["provision", "--catalog", str(catalog_file), "--data-root", str(tmp_path / "data")])

        with patch("otx.benchmark.catalog.provision_datasets", return_value={}) as mock_provision:
            rc = _cmd_provision(args)

        assert rc == 0
        mock_provision.assert_called_once()

    def test_provision_no_match_returns_zero(
        self,
        tmp_path: Path,
    ) -> None:
        # Write a catalog with no matching datasets
        cat_file = tmp_path / "empty_catalog.yaml"
        cat_file.write_text("version: 1\ndatasets: []\n")

        parser = _build_parser()
        args = parser.parse_args(
            [
                "provision",
                "--catalog",
                str(cat_file),
                "--data-root",
                str(tmp_path / "data"),
                "--dataset",
                "nonexistent",
            ]
        )
        rc = _cmd_provision(args)
        assert rc == 0


# ---------------------------------------------------------------------------
# Run sub-command
# ---------------------------------------------------------------------------


class TestCmdRun:
    @patch("otx.benchmark.runner.BenchmarkRunner")
    def test_dry_run_returns_zero(
        self,
        mock_runner_cls: MagicMock,
        catalog_file: Path,
        manifest_file: Path,
        tmp_path: Path,
    ) -> None:
        mock_runner = MagicMock()
        mock_runner.run.return_value = ([], [])
        mock_runner_cls.return_value = mock_runner

        parser = _build_parser()
        args = parser.parse_args(
            [
                "run",
                "--catalog",
                str(catalog_file),
                "--manifest",
                str(manifest_file),
                "--output-root",
                str(tmp_path / "results"),
                "--dry-run",
            ]
        )
        rc = _cmd_run(args)

        assert rc == 0
        mock_runner.run.assert_called_once()

    @patch("otx.benchmark.runner.BenchmarkRunner")
    def test_run_with_failures_returns_nonzero(
        self,
        mock_runner_cls: MagicMock,
        catalog_file: Path,
        manifest_file: Path,
        tmp_path: Path,
    ) -> None:
        from otx.benchmark.experiment import ExperimentResult

        failure = ExperimentResult(
            task="det",
            model="m",
            dataset="d",
            scenario="default",
            seed=0,
            success=False,
            error="boom",
        )
        mock_runner = MagicMock()
        mock_runner.run.return_value = ([], [failure])
        mock_runner_cls.return_value = mock_runner

        parser = _build_parser()
        args = parser.parse_args(
            [
                "run",
                "--catalog",
                str(catalog_file),
                "--manifest",
                str(manifest_file),
                "--output-root",
                str(tmp_path / "results"),
            ]
        )
        rc = _cmd_run(args)
        assert rc == 1


# ---------------------------------------------------------------------------
# main() entry point
# ---------------------------------------------------------------------------


class TestMain:
    @patch("otx.benchmark.cli.sys.exit")
    @patch("otx.benchmark.cli._cmd_provision", return_value=0)
    def test_main_dispatches_provision(
        self,
        mock_cmd: MagicMock,
        mock_exit: MagicMock,
        catalog_file: Path,
        tmp_path: Path,
    ) -> None:
        with patch(
            "sys.argv",
            ["prog", "provision", "--catalog", str(catalog_file), "--data-root", str(tmp_path)],
        ):
            main()
        mock_cmd.assert_called_once()
        mock_exit.assert_called_once_with(0)

    @patch("otx.benchmark.cli.sys.exit")
    @patch("otx.benchmark.cli._cmd_run", return_value=0)
    def test_main_dispatches_run(
        self,
        mock_cmd: MagicMock,
        mock_exit: MagicMock,
        catalog_file: Path,
        manifest_file: Path,
        tmp_path: Path,
    ) -> None:
        with patch(
            "sys.argv",
            [
                "prog",
                "run",
                "--catalog",
                str(catalog_file),
                "--manifest",
                str(manifest_file),
                "--output-root",
                str(tmp_path / "results"),
            ],
        ):
            main()
        mock_cmd.assert_called_once()
        mock_exit.assert_called_once_with(0)

    @patch("otx.benchmark.cli.sys.exit")
    @patch("otx.benchmark.cli._cmd_run", return_value=1)
    def test_main_propagates_nonzero_exit(
        self,
        mock_cmd: MagicMock,
        mock_exit: MagicMock,
        catalog_file: Path,
        manifest_file: Path,
        tmp_path: Path,
    ) -> None:
        with patch(
            "sys.argv",
            ["prog", "run", "--catalog", str(catalog_file), "--manifest", str(manifest_file)],
        ):
            main()
        mock_exit.assert_called_once_with(1)


# ---------------------------------------------------------------------------
# CLI filters pass-through to RunConfig
# ---------------------------------------------------------------------------


class TestCmdRunFilters:
    @patch("otx.benchmark.runner.BenchmarkRunner")
    def test_all_filters_passed_through(
        self,
        mock_runner_cls: MagicMock,
        catalog_file: Path,
        manifest_file: Path,
        tmp_path: Path,
    ) -> None:
        mock_runner = MagicMock()
        mock_runner.run.return_value = ([], [])
        mock_runner_cls.return_value = mock_runner

        parser = _build_parser()
        args = parser.parse_args(
            [
                "run",
                "--catalog",
                str(catalog_file),
                "--manifest",
                str(manifest_file),
                "--output-root",
                str(tmp_path / "results"),
                "--task",
                "detection",
                "--model",
                "yolox_s",
                "--dataset",
                "ds_a",
                "--priority",
                "core",
                "--size-tier",
                "tiny",
                "--scenario",
                "default",
                "--scenario-tag",
                "special",
                "--num-seeds",
                "3",
                "--max-epochs",
                "10",
                "--eval-upto",
                "export",
                "--accelerator",
                "xpu",
            ]
        )
        rc = _cmd_run(args)
        assert rc == 0

        # Verify RunConfig was constructed with correct parameters
        call_kwargs = mock_runner_cls.call_args
        config = call_kwargs[0][0] if call_kwargs[0] else call_kwargs[1].get("config")
        if config is None:
            # Constructed positionally
            config = mock_runner_cls.call_args[0][0]
        assert config.accelerator == "xpu"
        assert config.max_epochs == 10
        assert config.num_seeds == 3
        assert config.eval_upto == "export"

    @patch("otx.benchmark.runner.BenchmarkRunner")
    def test_provision_with_dataset_filter(
        self,
        mock_runner_cls: MagicMock,
        catalog_file: Path,
        tmp_path: Path,
    ) -> None:
        """Provision with --dataset filter should only provision matching datasets."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "provision",
                "--catalog",
                str(catalog_file),
                "--data-root",
                str(tmp_path / "data"),
                "--dataset",
                "ds_a",
            ]
        )
        with patch("otx.benchmark.catalog.provision_datasets", return_value={}) as mock_prov:
            rc = _cmd_provision(args)
        assert rc == 0
        mock_prov.assert_called_once()

    @patch("otx.benchmark.runner.BenchmarkRunner")
    def test_provision_with_size_tier_filter(
        self,
        mock_runner_cls: MagicMock,
        catalog_file: Path,
        tmp_path: Path,
    ) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "provision",
                "--catalog",
                str(catalog_file),
                "--data-root",
                str(tmp_path / "data"),
                "--size-tier",
                "tiny",
            ]
        )
        with patch("otx.benchmark.catalog.provision_datasets", return_value={}) as mock_prov:
            rc = _cmd_provision(args)
        assert rc == 0
        mock_prov.assert_called_once()


# ---------------------------------------------------------------------------
# --override / --train-kwarg / --rotation flags
# ---------------------------------------------------------------------------


class TestParseKeyValuePairs:
    def test_empty_returns_empty(self) -> None:
        assert _parse_key_value_pairs(None) == {}
        assert _parse_key_value_pairs([]) == {}

    def test_single_pair(self) -> None:
        result = _parse_key_value_pairs(["lr=0.01"])
        assert result == {"lr": "0.01"}

    def test_multiple_pairs(self) -> None:
        result = _parse_key_value_pairs(["lr=0.01", "precision=32"])
        assert result == {"lr": "0.01", "precision": "32"}

    def test_dotpath_key(self) -> None:
        result = _parse_key_value_pairs(["model.init_args.optimizer.init_args.lr=0.01"])
        assert result == {"model.init_args.optimizer.init_args.lr": "0.01"}

    def test_invalid_format_raises(self) -> None:
        import argparse as _argparse

        with pytest.raises(_argparse.ArgumentTypeError):
            _parse_key_value_pairs(["no_equals_sign"])


class TestNewCLIFlags:
    def test_override_flag_parsed(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "run",
                "--override",
                "model.init_args.optimizer.init_args.lr=0.01",
                "batch_size=16",
            ]
        )
        assert args.override == ["model.init_args.optimizer.init_args.lr=0.01", "batch_size=16"]

    def test_train_kwarg_flag_parsed(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "run",
                "--train-kwarg",
                "precision=32",
            ]
        )
        assert args.train_kwarg == ["precision=32"]

    def test_rotation_group_parsed(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "--rotation-group", "2"])
        assert args.rotation_group == 2

    def test_no_rotation_parsed(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "--no-rotation"])
        assert args.no_rotation is True

    def test_defaults_for_new_flags(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run"])
        assert args.override is None
        assert args.train_kwarg is None
        assert args.rotation_group is None
        assert args.no_rotation is False
