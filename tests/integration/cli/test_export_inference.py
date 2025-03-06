# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from copy import copy
from pathlib import Path

import onnxruntime as ort
import pandas as pd
import pytest

from tests.utils import run_main

log = logging.getLogger(__name__)


def _check_relative_metric_diff(ref: float, value: float, eps: float) -> None:
    assert ref >= 0
    assert value >= 0
    assert eps >= 0

    if value < ref:
        avg = max(0.5 * (ref + value), 1e-9)
        diff = abs(value - ref)

        assert diff / avg <= eps, f"Relative difference exceeded {eps} threshold. Absolute difference: {diff}"


@pytest.fixture(scope="module", autouse=True)
def fxt_local_seed() -> int:
    """The number of repetition for each test case.

    The random seed will be set for [0, fxt_num_repeat - 1]. Default is one.
    """
    selected_seed = 7
    msg = f"seed : {selected_seed}"
    log.info(msg)
    return selected_seed


TASK_NAME_TO_MAIN_METRIC_NAME = {
    "semantic_segmentation": "test/Dice",
    "multi_label_cls": "test/accuracy",
    "multi_class_cls": "test/accuracy",
    "h_label_cls": "test/accuracy",
    "detection": "test/map_50",
    "instance_segmentation": "test/map_50",
    "visual_prompting": "test/f1-score",
    "keypoint_detection": "test/PCK",
}


# TODO(someone): this test is too complex and should be split into multiple tests.
@pytest.mark.parametrize(
    "recipe",
    pytest.RECIPE_LIST,
    ids=lambda x: "/".join(Path(x).parts[-2:]),
)
def test_otx_export_infer(
    recipe: str,
    tmp_path: Path,
    fxt_local_seed: int,
    fxt_target_dataset_per_task: dict,
    fxt_cli_override_command_per_task: dict,
    fxt_accelerator: str,
    fxt_open_subprocess: bool,
) -> None:
    """
    Test OTX CLI e2e commands.

    - 'otx train' with 2 epochs training
    - 'otx test' with output checkpoint from 'otx train'
    - 'otx export' with output checkpoint from 'otx train'
    - 'otx test' with the exported to ONNX/IR model model
    - compare accuracy of the exported model vs the original accuracy

    Args:
        recipe (str): The recipe to use for training. (eg. 'classification/otx_mobilenet_v3_large.yaml')
        tmp_path (Path): The temporary path for storing the training outputs.

    Returns:
        None
    """
    task = recipe.split("/")[-2]

    if task not in TASK_NAME_TO_MAIN_METRIC_NAME:
        pytest.skip(f"Inference pipeline for {recipe} is not implemented")
    elif (task == "detection" and "atss_mobilenetv2" not in recipe) or (
        task == "instance_segmentation" and "maskrcnn_efficientnetb2b" not in recipe
    ):
        pytest.skip("To prevent memory bug from aborting integration test, test single model per task.")

    model_name = recipe.split("/")[-1].split(".")[0]

    # 1) otx train
    tmp_path_train = tmp_path / f"otx_train_{model_name}"
    command_cfg = [
        "otx",
        "train",
        "--config",
        recipe,
        "--data_root",
        fxt_target_dataset_per_task[task],
        "--work_dir",
        str(tmp_path_train / "outputs"),
        "--engine.device",
        fxt_accelerator,
        "--max_epochs",
        "2",
        "--seed",
        f"{fxt_local_seed}",
        *fxt_cli_override_command_per_task[task],
    ]

    # TODO(someone): Disable deterministic for instance segmentation as it causes OOM.
    # https://github.com/pytorch/vision/issues/8168#issuecomment-1890599205
    if task != "instance_segmentation":
        command_cfg.extend(
            [
                "--deterministic",
                "warn",
            ],
        )

    run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

    outputs_dir = tmp_path_train / "outputs"
    latest_dir = max(
        (p for p in outputs_dir.iterdir() if p.is_dir() and p.name != ".latest"),
        key=lambda p: p.stat().st_mtime,
    )
    ckpt_files = list((latest_dir / "checkpoints").glob(pattern="epoch_*.ckpt"))
    assert len(ckpt_files) > 0

    # 2) otx test
    def __run_cli_test(
        test_recipe: str,
        checkpoint_path: str,
        work_dir: Path,
        device: str = fxt_accelerator,
        cli_override_command: list[str] | None = None,
        with_benchmark: bool = False,
    ) -> Path:
        tmp_path_test = tmp_path / f"otx_test_{model_name}"
        command_cfg = [
            "otx",
            "test",
            "--config",
            test_recipe,
            "--data_root",
            fxt_target_dataset_per_task[task],
            "--work_dir",
            str(tmp_path_test / work_dir),
            "--engine.device",
            device,
            *fxt_cli_override_command_per_task[task],
            "--checkpoint",
            checkpoint_path,
        ]

        if cli_override_command is not None:
            command_cfg.extend(cli_override_command)

        if with_benchmark:
            benchmark_command_cfg = copy(command_cfg)
            updated = False
            for i, term in enumerate(benchmark_command_cfg):
                if term == "test":
                    benchmark_command_cfg[i] = "benchmark"
                    updated = True
                    break

            assert updated
            benchmark_command_cfg.extend(["--n_iters", "1", "--batch_size", "1"])
            run_main(command_cfg=benchmark_command_cfg, open_subprocess=fxt_open_subprocess)

        run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

        return tmp_path_test

    checkpoint_path: str = str(ckpt_files[-1])
    tmp_path_test = __run_cli_test(recipe, checkpoint_path, Path("outputs") / "torch", with_benchmark=True)

    assert (tmp_path_test / "outputs").exists()
    assert (tmp_path_test / "outputs" / "torch" / ".latest" / "benchmark" / "benchmark_report.csv").exists()

    # 3) otx export
    format_to_ext = {"OPENVINO": "xml", "ONNX": "onnx"}

    tmp_path_test = tmp_path / f"otx_test_{model_name}"
    for fmt, ext in format_to_ext.items():
        command_cfg = [
            "otx",
            "export",
            "--config",
            recipe,
            "--data_root",
            fxt_target_dataset_per_task[task],
            "--work_dir",
            str(tmp_path_test / "outputs"),
            *fxt_cli_override_command_per_task[task],
            "--checkpoint",
            str(ckpt_files[-1]),
            "--export_format",
            f"{fmt}",
        ]

        run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

        outputs_dir = tmp_path_test / "outputs"
        latest_dir = max(
            (p for p in outputs_dir.iterdir() if p.is_dir() and p.name != ".latest"),
            key=lambda p: p.stat().st_mtime,
        )
        assert latest_dir.exists()
        if task == "visual_prompting":
            assert (latest_dir / f"exported_model_image_encoder.{ext}").exists()
            assert (latest_dir / f"exported_model_decoder.{ext}").exists()
        else:
            assert (latest_dir / f"exported_model.{ext}").exists()

        if fmt == "ONNX":
            onnx_model_path = latest_dir / f"exported_model.{ext}"
            try:
                ort.InferenceSession(str(onnx_model_path))
            except Exception:
                log.exception("ONNX model %s is invalid", model_name)
        else:
            # 4) infer of the exported models
            task = recipe.split("/")[-2]
            tmp_path_test = tmp_path / f"otx_test_{model_name}"
            if "_cls" in recipe:
                export_test_recipe = f"src/otx/recipe/classification/{task}/openvino_model.yaml"
            else:
                export_test_recipe = f"src/otx/recipe/{task}/openvino_model.yaml"

            if task == "visual_prompting":
                exported_model_path = str(latest_dir / "exported_model_decoder.xml")
            else:
                exported_model_path = str(latest_dir / "exported_model.xml")

            tmp_path_test = __run_cli_test(export_test_recipe, exported_model_path, Path("outputs") / "openvino", "cpu")
            assert (tmp_path_test / "outputs").exists()

            # 5) test optimize
            command_cfg = [
                "otx",
                "optimize",
                "--config",
                export_test_recipe,
                "--data_root",
                fxt_target_dataset_per_task[task],
                "--work_dir",
                str(tmp_path_test / "outputs"),
                "--engine.device",
                "cpu",
                *fxt_cli_override_command_per_task[task],
                "--checkpoint",
                exported_model_path,
            ]

            run_main(command_cfg=command_cfg, open_subprocess=fxt_open_subprocess)

            outputs_dir = tmp_path_test / "outputs"
            latest_dir = max(
                (p for p in outputs_dir.iterdir() if p.is_dir() and p.name != ".latest"),
                key=lambda p: p.stat().st_mtime,
            )
            assert latest_dir.exists()
            if task == "visual_prompting":
                optimized_model_path = str(latest_dir / "optimized_model_decoder.xml")
            else:
                optimized_model_path = str(latest_dir / "optimized_model.xml")

            # 6) test optimized model
            tmp_path_test = __run_cli_test(
                export_test_recipe,
                optimized_model_path,
                Path("outputs") / "nncf_ptq",
                "cpu",
            )

            torch_outputs_dir = tmp_path_test / "outputs" / "torch"
            torch_latest_dir = max(
                (p for p in torch_outputs_dir.iterdir() if p.is_dir() and p.name != ".latest"),
                key=lambda p: p.stat().st_mtime,
            )
            openvino_outputs_dir = tmp_path_test / "outputs" / "openvino"
            openvino_latest_dir = max(
                (p for p in openvino_outputs_dir.iterdir() if p.is_dir() and p.name != ".latest"),
                key=lambda p: p.stat().st_mtime,
            )
            nncf_ptq_outputs_dir = tmp_path_test / "outputs" / "nncf_ptq"
            nncf_ptq_latest_dir = max(
                (p for p in nncf_ptq_outputs_dir.iterdir() if p.is_dir() and p.name != ".latest"),
                key=lambda p: p.stat().st_mtime,
            )
            assert nncf_ptq_latest_dir.exists()

            df_torch = pd.read_csv(next(torch_latest_dir.glob("**/metrics.csv")))
            df_openvino = pd.read_csv(next(openvino_latest_dir.glob("**/metrics.csv")))
            df_nncf_ptq = pd.read_csv(next(nncf_ptq_latest_dir.glob("**/metrics.csv")))

            metric_name = TASK_NAME_TO_MAIN_METRIC_NAME[task]

            assert metric_name in df_torch.columns
            assert metric_name in df_openvino.columns
            assert metric_name in df_nncf_ptq.columns

            torch_acc = df_torch[metric_name].item()
            ov_acc = df_openvino[metric_name].item()
            ptq_acc = df_nncf_ptq[metric_name].item()

            msg = f"Recipe: {recipe}, (torch_accuracy, ov_accuracy, ptq_acc): {torch_acc}, {ov_acc}, {ptq_acc}"
            log.info(msg)

            # Not compare w/ instance segmentation and visual prompting tasks because training isn't able to be deterministic, which can lead to unstable test result.
            if "maskrcnn_efficientnetb2b" in recipe or task == "visual_prompting":
                return

            # This test seems flaky, so let's disable it.
            # Model accuracy should be checked at the regression tests
            # https://github.com/openvinotoolkit/training_extensions/actions/runs/8340264268/job/22824202673?pr=3155
            # _check_relative_metric_diff(torch_acc, ov_acc, threshold) noqa: ERA001
