# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX benchmark runner."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from time import time
from typing import Any, Literal

import pandas as pd

from otx.backend.native.cli.utils import RECIPE_PATH
from otx.backend.native.engine import OTXEngine
from otx.backend.openvino.engine import OVEngine
from otx.types.task import OTXTaskType
from tests.perf_v2 import CRITERIA_COLLECTIONS, DATASET_COLLECTIONS, MODEL_COLLECTIONS, summary
from tests.perf_v2.utils import (
    Criterion,
    DatasetInfo,
    ModelInfo,
    RunTestType,
    SubCommand,
    build_tags,
    current_date_str,
    get_parser,
    get_version_tags,
    load_result,
)

logger = logging.getLogger(__name__)

FOLDER_MAPPINGS = {
    OTXTaskType.MULTI_CLASS_CLS: RECIPE_PATH / "classification" / "multi_class_cls",
    OTXTaskType.MULTI_LABEL_CLS: RECIPE_PATH / "classification" / "multi_label_cls",
    OTXTaskType.H_LABEL_CLS: RECIPE_PATH / "classification" / "h_label_cls",
    OTXTaskType.DETECTION: RECIPE_PATH / "detection",
    OTXTaskType.ROTATED_DETECTION: RECIPE_PATH / "rotated_detection",
    OTXTaskType.SEMANTIC_SEGMENTATION: RECIPE_PATH / "semantic_segmentation",
    OTXTaskType.INSTANCE_SEGMENTATION: RECIPE_PATH / "instance_segmentation",
    OTXTaskType.ANOMALY: RECIPE_PATH / "anomaly",
    OTXTaskType.ANOMALY_CLASSIFICATION: RECIPE_PATH / "anomaly_classification",
    OTXTaskType.ANOMALY_SEGMENTATION: RECIPE_PATH / "anomaly_segmentation",
    OTXTaskType.ANOMALY_DETECTION: RECIPE_PATH / "anomaly_detection",
    OTXTaskType.KEYPOINT_DETECTION: RECIPE_PATH / "keypoint_detection",
}


def task_benchmark_dataset(task: OTXTaskType) -> dict[str, DatasetInfo]:
    test_cases = DATASET_COLLECTIONS[task]
    return {test_case.name: test_case for test_case in test_cases}


def task_benchmark_models(task: OTXTaskType) -> dict[str, ModelInfo]:
    model_info_list = MODEL_COLLECTIONS[task]
    return {model.name: model for model in model_info_list}


class AggregateError(Exception):
    def __init__(self, errors):
        error_messages = []
        for seed, error in errors:
            error_messages.append(f"Seed {seed}: {error}")
        error_message = "\n".join(error_messages)

        super().__init__(f"Exceptions occurred in the following seeds:\n{error_message}")


class Benchmark:
    """Benchmark runner for OTX2.x.

    Args:
        data_root (str): Path to the root of dataset directories. Defaults to './data'.
        output_root (str): Output root dirctory for logs and results. Defaults to './otx-benchmark'.
        num_epoch (int): Overrides the per-model default number of epoch settings.
            Defaults to 0, which means no overriding.
        eval_upto (str): The last serial operation to evaluate. Choose one of ('train', 'export', 'optimize').
            Operations include the preceeding ones.
            e.x) Eval up to 'optimize': train -> eval -> export -> eval -> optimize -> eval
            Default to 'train'.
        tags (dict, optional): Key-values pair metadata for the experiment.
        dry_run (bool): Whether to just print the OTX command without execution. Defaults to False.
        deterministic (bool): Whether to turn on deterministic training mode. Defaults to False.
        accelerator (str): Accelerator device on which to run benchmark. Defaults to gpu.
        reference_results (pd.DataFrame): Reference benchmark results for performance checking.
        resume_from (Path | None, optional):
            Previous performance directory to load. If training was already done in previous performance test,
            training is skipped and refer previous result.
        test_only (Literal["all", "train", "export", "optimize"] | None):
            Execute test only when resume argument is given. If necessary files are not found in resume directory,
            necessary operations can be executed. Defaults to None.
    """

    def __init__(
        self,
        data_root: Path = Path("data"),
        output_root: Path = Path("otx-benchmark"),
        num_epoch: int = 0,
        eval_upto: str = "train",
        tags: dict[str, str] | None = None,
        dry_run: bool = False,
        deterministic: bool = False,
        accelerator: str = "gpu",
        reference_results: pd.DataFrame | None = None,
        resume_from: Path | None = None,
        test_only: Literal["all", "train", "export", "optimize"] | None = None,
    ):
        self.data_root = data_root
        self.output_root = output_root
        self.num_epoch = num_epoch
        self.eval_upto = eval_upto
        self.tags = tags or {}
        self.dry_run = dry_run
        self.deterministic = deterministic
        self.accelerator = accelerator
        self.reference_results = reference_results
        self.resume_from = resume_from
        if (test_only == "export" and eval_upto == "train") or (
            test_only == "optimize" and eval_upto in ["train", "export"]
        ):
            msg = "test_only should be set to previous otx command than eval_upto."
            raise ValueError(msg)
        self.test_only = test_only

    def train(
        self,
        model_info: ModelInfo,
        dataset_info: DatasetInfo,
        sub_work_dir: Path,
        seed: int,
    ) -> float:
        """Train model with given dataset and return the total time.

        Args:
            model_info (ModelInfo):
            dataset_info (DatasetInfo):
            sub_work_dir (Path):
            seed (int):

        Returns:
            float: Total time for training
        """

        engine = self._initialize_engine(
            model_info=model_info,
            dataset_info=dataset_info,
            work_dir=sub_work_dir / SubCommand.TRAIN.value,
        )

        kwargs = {}
        if extra_overrides := dataset_info.extra_overrides:
            for key, value in extra_overrides.get("train", {}).items():
                kwargs[key] = value
        kwargs["seed"] = seed
        kwargs["deterministic"] = self.deterministic
        if self.num_epoch > 0:
            kwargs["max_epochs"] = self.num_epoch

        # ======Train======
        start_time = time()
        engine.train(**kwargs)
        total_time = time() - start_time
        # =================

        self._rename_raw_data(
            work_dir=Path(engine.work_dir),
            replace_map={"train_": "train/", "{pre}": "training:"},
        )
        del engine
        return total_time

    def test(
        self,
        model_info: ModelInfo,
        dataset_info: DatasetInfo,
        sub_work_dir: Path | str,
        tags: dict[str, str],
        criteria: list[Criterion],
        checkpoint: Path | str | None = None,
        what2test: RunTestType = RunTestType.TORCH,
    ) -> None:
        """Test model with given dataset on the given checkpoint.

        There are 3 type of tests: torch, export, optimize. Each test has different checkpoint and output directory.
        * Test Directory For PyTorch: {sub_work_dir}/test/torch
        * Test Directory For OV Export: {sub_work_dir}/test/export
        * Test Directory For POT Optimize: {sub_work_dir}/test/optimize

        Args:
            model_info (ModelInfo): Information of the model to test
            dataset_info (DatasetInfo): Information of the dataset to test
            sub_work_dir (Path | str): Sub work directory
            tags (dict[str, str]): Information useful for excel/csv output
            criteria (list[Criterion]): Criteria to check results
            checkpoint (Path | str | None, optional): Checkpoint path. Defaults to None.
            what2test (RunTestType, optional): indicates which model to test. Defaults to RunTestType.TORCH.
        """
        test_type = what2test.value
        if self.test_only not in ["all", test_type, None]:
            return

        work_dir = sub_work_dir / SubCommand.TEST.value / test_type
        if what2test == RunTestType.TORCH:
            engine = self._initialize_engine(
                model_info=model_info,
                dataset_info=dataset_info,
                work_dir=work_dir,
            )
        else:
            engine = OVEngine(
                work_dir=work_dir,
                data=self.data_root / dataset_info.path,
                model=checkpoint,
            )

        kwargs = {}
        if extra_overrides := dataset_info.extra_overrides:
            for key, value in extra_overrides.get("test", {}).items():
                kwargs[key] = value
        kwargs.update(kwargs)
        kwargs.pop("checkpoint", None)  # Remove checkpoint

        # ======Test=======
        start_time = time()
        engine.test(
            checkpoint=checkpoint,
            **kwargs,
        )
        total_time = time() - start_time

        # NOTE: This is a very rough estimation of latency.
        # It is calculated by dividing the total time by the number of samples.
        latency = total_time / len(engine.datamodule.subsets["test"])
        extra_metrics = {
            f"{test_type}:test/e2e_time": total_time,
            f"{test_type}:test/latency": latency,
        }
        # =================

        self._rename_raw_data(
            work_dir=Path(engine.work_dir),
            replace_map={"test_": "test/", "{pre}": f"{test_type}:", "image_F1Score": "test/image_F1Score"},
        )
        self._log_metrics(
            work_dir=Path(engine.work_dir),
            tags=tags,
            criteria=criteria,
            extra_metrics=extra_metrics,
        )
        del engine

    def export(
        self,
        model_info: ModelInfo,
        dataset_info: DatasetInfo,
        sub_work_dir: Path,
    ) -> None:
        engine = self._initialize_engine(
            model_info=model_info,
            dataset_info=dataset_info,
            work_dir=sub_work_dir / SubCommand.EXPORT.value,
        )

        kwargs = {}
        if extra_overrides := dataset_info.extra_overrides:
            for key, value in extra_overrides.get("export", {}).items():
                kwargs[key] = value

        ckpt_path = sub_work_dir / "train" / "best_checkpoint.ckpt"
        if not ckpt_path.exists():
            msg = f"Checkpoint file not found: {ckpt_path}"
            raise FileNotFoundError(msg)

        kwargs.pop("checkpoint", None)  # Remove checkpoint
        engine.export(
            checkpoint=ckpt_path,
            **kwargs,
        )
        del engine

    def optimize(
        self,
        dataset_info: DatasetInfo,
        sub_work_dir: Path,
        exported_model_path: Path,
    ):
        engine = OVEngine(
            work_dir=sub_work_dir / SubCommand.OPTIMIZE.value,
            data=self.data_root / dataset_info.path,
            model=exported_model_path,
        )

        kwargs = {}
        if extra_overrides := dataset_info.extra_overrides:
            for key, value in extra_overrides.get("optimize", {}).items():
                kwargs[key] = value

        kwargs.pop("checkpoint", None)  # Remove checkpoint

        # ======Optimize=======
        start_time = time()

        engine.optimize(
            checkpoint=exported_model_path,
            **kwargs,
        )
        del engine
        total_time = time() - start_time

        # OTX does not create metrics.cvs during optimization,
        # So we are manually write optimize:e2e_time to csv.
        data_frame = pd.DataFrame({"optimize:e2e_time": [total_time]})
        data_frame.to_csv(sub_work_dir / f"{SubCommand.OPTIMIZE.value}/metrics.csv", index=False)
        # =================

    def _initialize_engine(
        self,
        model_info: ModelInfo,
        dataset_info: DatasetInfo,
        work_dir: Path,
    ) -> OTXEngine:
        """Initialise engine with given model and dataset settings.

        Args:
            model_info (ModelInfo): Target model settings
            dataset_info (DatasetInfo): Target dataset settings
            work_dir (Path): Sub work directory

        Returns:
            Engine: Initialised engine
        """

        return OTXEngine.from_config(
            config_path=FOLDER_MAPPINGS[model_info.task] / (model_info.name + ".yaml"),
            data_root=self.data_root / dataset_info.path,
            work_dir=work_dir,
            device=self.accelerator,
        )

    def run(
        self,
        model_info: ModelInfo,
        dataset_info: DatasetInfo,
        seed: int,
        criteria: list[Criterion],
    ) -> pd.DataFrame | None:
        """Run configured benchmark with given dataset and model and return the result.

        Args:
            model_info (ModelInfo): Model to benchmark cotaining task, name, and category
            dataset_info (DatasetInfo): Dataset to benchmark containing name, path, group, and num_repeat
            criteria (list[Criterion]): Criteria to check results

        Retruns:
            pd.DataFrame | None: Table with benchmark metrics
        """

        run_name = f"{model_info.name}/{dataset_info.name}"
        logger.info(f"{run_name = }")
        work_dir = self.output_root / run_name

        tags = {
            "task": model_info.task,
            "data_group": dataset_info.group,
            "model": model_info.name,
            "data": dataset_info.name,
            **self.tags,
        }

        exceptions = []
        try:
            sub_work_dir = work_dir / str(seed)
            tags["seed"] = str(seed)

            # Check if operation was already done in previous run
            # If so, copy the previous operation directory
            copied_ops_dir = self._prepare_resume(sub_work_dir)

            # Run training if not in resume operation
            if "train" not in copied_ops_dir:
                if sub_work_dir.exists():
                    shutil.rmtree(sub_work_dir)

                e2e_train_time = self.train(
                    model_info=model_info,
                    dataset_info=dataset_info,
                    sub_work_dir=sub_work_dir,
                    seed=seed,
                )

                self._log_metrics(
                    work_dir=sub_work_dir / SubCommand.TRAIN.value,
                    tags=tags,
                    criteria=criteria,
                    extra_metrics={
                        "training:e2e_time": e2e_train_time,
                    },
                )

            if "train" in copied_ops_dir:
                self._log_metrics(
                    work_dir=sub_work_dir / SubCommand.TRAIN.value,
                    tags=tags,
                    criteria=criteria,
                )

            self.test(
                model_info=model_info,
                dataset_info=dataset_info,
                sub_work_dir=sub_work_dir,
                tags=tags,
                criteria=criteria,
                checkpoint=sub_work_dir / "train" / "best_checkpoint.ckpt",
                what2test=RunTestType.TORCH,
            )

            # Export & test
            if self.eval_upto in ["export", "optimize"]:
                if "export" not in copied_ops_dir:
                    self.export(
                        model_info=model_info,
                        dataset_info=dataset_info,
                        sub_work_dir=sub_work_dir,
                    )

                exported_model_path = sub_work_dir / "export" / "exported_model.xml"
                if not exported_model_path.exists():
                    exported_model_path = sub_work_dir / ".latest" / "export" / "exported_model_decoder.xml"

                # Test OpenVINO exported model
                self.test(
                    model_info=model_info,
                    dataset_info=dataset_info,
                    sub_work_dir=sub_work_dir,
                    tags=tags,
                    criteria=criteria,
                    checkpoint=exported_model_path,
                    what2test=RunTestType.EXPORT,
                )

            # Optimize & test
            if self.eval_upto == "optimize":
                if "optimize" not in copied_ops_dir:
                    self.optimize(
                        dataset_info=dataset_info,
                        sub_work_dir=sub_work_dir,
                        exported_model_path=exported_model_path,
                    )
                    self._log_metrics(
                        work_dir=sub_work_dir / SubCommand.OPTIMIZE.value,
                        tags=tags,
                        criteria=criteria,
                    )

                optimized_model_path = sub_work_dir / "optimize" / "optimized_model.xml"

                self.test(
                    model_info=model_info,
                    dataset_info=dataset_info,
                    sub_work_dir=sub_work_dir,
                    tags=tags,
                    criteria=criteria,
                    checkpoint=optimized_model_path,
                    what2test=RunTestType.OPTIMIZE,
                )
        except Exception as e:
            exceptions.append((seed, str(e)))

        if exceptions:
            # Raise the custom exception with all collected errors
            raise AggregateError(exceptions)

        result = load_result(work_dir)
        if result is None:
            return None
        result = summary.average(result, keys=["task", "model", "data_group", "data"])  # Average out seeds
        return result.set_index(["task", "model", "data_group", "data"])

    def _prepare_resume(self, work_dir: Path) -> list[str]:
        return [subcommand for subcommand in ("train", "export", "optimize") if (work_dir / subcommand).exists()]

    def _find_resume_directory(self, tags: dict[str, str]) -> Path | None:
        for csv_file in self.resume_from.rglob("benchmark.raw.csv"):
            if csv_file.parent.name == ".latest":
                continue
            raw_data = pd.read_csv(csv_file)
            if all(  # check meta info is same
                str(raw_data.iloc[0].get(key, "NOT_IN_CSV")) == tags.get(key, "NOT_IN_TAG")
                for key in ["data_group", "data", "model", "task", "seed"]
            ):
                return csv_file.parent.parent
        return None

    def _log_metrics(
        self,
        work_dir: Path,
        tags: dict[str, str],
        criteria: list[Criterion],
        extra_metrics: dict[str, Any] | None = None,
    ) -> None:
        """Log metrics and tags to csv file.

        Args:
            work_dir (Path): work directory
            tags (dict[str, str]): OTX metadata (ie. task, model, data_group, data, seed, etc)
            criteria (list[Criterion]): task criteria
            extra_metrics (dict[str, Any] | None, optional): extra metrics to be logged. Defaults to None.
        """

        if not work_dir.exists():
            return

        # Load raw metrics
        csv_files = work_dir.glob("**/metrics.csv")
        raw_data = [pd.read_csv(csv_file) for csv_file in csv_files]
        raw_data = pd.concat(raw_data, ignore_index=True)
        if extra_metrics:
            for k, v in extra_metrics.items():
                raw_data[k] = v

        # Summarize
        metrics = []
        for criterion in criteria:
            if criterion.name not in raw_data:
                continue
            column = raw_data[criterion.name].dropna()
            if len(column) == 0:
                continue
            if criterion.summary == "mean":
                value = column[min(1, len(column) - 1) :].mean()  # Drop 1st epoch if possible
            elif criterion.summary == "max":
                value = column.max()
            elif criterion.summary == "min":
                value = column.min()
            else:
                value = 0.0
            metrics.append(pd.Series([value], name=criterion.name))
        if len(metrics) == 0:
            return
        metrics = pd.concat(metrics, axis=1)

        # Write csv w/ tags
        for k, v in tags.items():
            metrics[k] = v

        if (work_dir / "benchmark.raw.csv").exists():
            # If raw csv already exists, combine with existing metrics
            existing_metrics = pd.read_csv(work_dir / "benchmark.raw.csv")
            for col, val in metrics.items():
                if col not in existing_metrics.columns:
                    existing_metrics[col] = val
                    print(f"Adding new column {col} to existing metrics.")

            existing_metrics.to_csv(work_dir / "benchmark.raw.csv", index=False)
        else:
            # If raw csv does not exist, create new one
            metrics.to_csv(work_dir / "benchmark.raw.csv", index=False)

    def _rename_raw_data(self, work_dir: Path, replace_map: dict[str, str]) -> None:
        """Rename columns in the metrics.csv files based on the provided replacements.

        Args:
            work_dir (Path): work directory
            replace_map (dict[str, str]): Replacement map
        """

        def _rename_col(col_name: str) -> str:
            for src_str, dst_str in replace_map.items():
                if src_str == "{pre}":
                    if not col_name.startswith(dst_str):
                        col_name = dst_str + col_name
                elif src_str == "{post}":
                    if not col_name.endswith(dst_str):
                        col_name = col_name + dst_str
                else:
                    col_name = col_name.replace(src_str, dst_str)
            return col_name

        csv_files = work_dir.glob("**/metrics.csv")
        for csv_file in csv_files:
            data = pd.read_csv(csv_file)
            data = data.rename(columns=_rename_col)  # Column names
            data = data.replace(replace_map)  # Values
            data.to_csv(csv_file, index=False)

    def check(self, result: pd.DataFrame, criteria: list[Criterion]):
        """Check result w.r.t. reference data.

        Args:
            result (pd.DataFrame): Result data frame
            criteria (list[Criterion]): Criteria to check results
        """
        if result is None:
            print("[Check] No results loaded. Skipping result checking.")
            return

        if self.reference_results is None:
            print("[Check] No benchmark references loaded. Skipping result checking.")
            return

        for key, result_entry in result.iterrows():
            if key not in self.reference_results.index:
                print(f"[Check] No benchmark reference for {key} loaded. Skipping result checking.")
                continue
            target_entry = self.reference_results.loc[key]
            if isinstance(target_entry, pd.DataFrame):
                # Match num_repeat of result and target
                result_seed_average = result_entry["seed"]
                result_num_repeat = 2 * result_seed_average + 1  # (0+1+2+3+4)/5 = 2.0 -> 2*2.0+1 = 5
                target_entry = target_entry.query(f"seed < {result_num_repeat}")
                target_entry = target_entry.mean(numeric_only=True)  # N-row pd.DataFrame to pd.Series

            for criterion in criteria:
                criterion(result_entry, target_entry)


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument(
        "--model",
        type=str,
        help="Choose a model from the predefined list.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Choose a dataset from the predefined list.",
    )
    args = parser.parse_args()

    current_date = current_date_str()
    version_tags = get_version_tags(current_date)
    tags = build_tags(args, version_tags)

    model_info = task_benchmark_models(args.task)[args.model]
    dataset_info = task_benchmark_dataset(args.task)[args.dataset]
    criteria = CRITERIA_COLLECTIONS[args.task]

    # Here, you can instantiate your Benchmark with the parameters from config.
    benchmark = Benchmark(
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
        num_epoch=args.num_epoch,
        eval_upto=args.eval_upto,
        tags=tags,
        dry_run=args.dry_run,
        deterministic=(
            False if args.deterministic is None else {"true": True, "false": False, "warn": "warn"}[args.deterministic]
        ),
        accelerator=args.device,
        reference_results=None,  # or load your benchmark reference data if available
        resume_from=Path(args.resume_from) if args.resume_from else None,
        test_only=args.test_only,
    )

    result = benchmark.run(
        model_info=model_info,
        dataset_info=dataset_info,
        seed=args.seed,
        criteria=criteria,
    )
    benchmark.check(
        result=result,
        criteria=criteria,
    )
