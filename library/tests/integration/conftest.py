# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import importlib
import inspect
from enum import Enum
from pathlib import Path

import pytest

from otx.types.task import OTXTaskType


class ModelStatus(str, Enum):
    """Enum for model status used to categorize models for integration testing."""

    SPEED = "speed"
    BALANCE = "balance"
    ACCURACY = "accuracy"


# Mapping of model category recipes. This mirrors the BALANCE/SPEED/ACCURACY entries
# from the application's TEMPLATE_ID_MAPPING so that --run-category-only selects the
# same subset of models for integration testing.
_CATEGORY_RECIPES: dict[str, ModelStatus] = {
    # MULTI_CLASS_CLS
    "classification/multi_class_cls/deit_tiny.yaml": ModelStatus.BALANCE,
    "classification/multi_class_cls/dino_v2.yaml": ModelStatus.ACCURACY,
    "classification/multi_class_cls/mobilenet_v3_large.yaml": ModelStatus.SPEED,
    # DETECTION
    "detection/yolox_s.yaml": ModelStatus.SPEED,
    "detection/deim_dfine_m.yaml": ModelStatus.BALANCE,
    "detection/deim_dfine_l.yaml": ModelStatus.ACCURACY,
    # INSTANCE_SEGMENTATION
    "instance_segmentation/rfdetr_seg_small.yaml": ModelStatus.SPEED,
    "instance_segmentation/rfdetr_seg_medium.yaml": ModelStatus.BALANCE,
    "instance_segmentation/rfdetr_seg_xlarge.yaml": ModelStatus.ACCURACY,
}


@pytest.fixture(scope="module", autouse=True)
def fxt_open_subprocess(request: pytest.FixtureRequest) -> bool:
    """Open subprocess for each CLI integration test case.

    This option can be used for easy memory management
    while running consecutive multiple tests (default: false).
    """
    return request.config.getoption("--open-subprocess", False)


def find_recipe_folder(base_path: Path, folder_name: str) -> Path:
    """
    Find the folder with the given name within the specified base path.

    Args:
        base_path (Path): The base path to search within.
        folder_name (str): The name of the folder to find.

    Returns:
        Path: The path to the folder.
    """
    for folder_path in base_path.rglob(folder_name):
        if folder_path.is_dir():
            return folder_path
    msg = f"Folder {folder_name} not found in {base_path}."
    raise FileNotFoundError(msg)


def get_task_list(task: str) -> list[OTXTaskType]:
    if task == "all":
        tasks = list(OTXTaskType)
    elif task == "multi_class_cls":
        tasks = [OTXTaskType.MULTI_CLASS_CLS]
    elif task == "multi_label_cls":
        tasks = [OTXTaskType.MULTI_LABEL_CLS]
    elif task == "h_label_cls":
        tasks = [OTXTaskType.H_LABEL_CLS]
    elif task == "classification":
        tasks = [OTXTaskType.MULTI_CLASS_CLS, OTXTaskType.MULTI_LABEL_CLS, OTXTaskType.H_LABEL_CLS]
    elif task == "keypoint_detection":
        tasks = [OTXTaskType.KEYPOINT_DETECTION]
    else:
        tasks = [OTXTaskType(task.upper())]
    return tasks


def get_model_category_list(task: str, recipe_path: Path) -> list[str]:
    """
    Retrieve the list of category model recipes (BALANCE, SPEED, ACCURACY).

    Uses the local _CATEGORY_RECIPES mapping to select only the representative
    models for each category, matching the application's TEMPLATE_ID_MAPPING.

    For classification tasks, multi_class_cls recipes are also mapped to
    multi_label_cls and h_label_cls variants when those tasks are requested.

    Args:
        task (str): The task for which to retrieve model categories.
        recipe_path (Path): The root recipe directory.

    Returns:
        list[str]: A list of recipe paths.
    """
    task_list = get_task_list(task.lower())
    recipes = []

    for relative_path in _CATEGORY_RECIPES:
        full_path = recipe_path / relative_path

        # Extract task from the path (e.g. "multi_class_cls" from ".../multi_class_cls/deit_tiny.yaml")
        task_from_path = OTXTaskType(str(full_path).split("/")[-2].upper())
        if task_from_path in task_list:
            recipes.append(str(full_path))

        if task_from_path == OTXTaskType.MULTI_CLASS_CLS:
            # Add multi_label_cls and h_label_cls configs as well if they are in the list
            if OTXTaskType.MULTI_LABEL_CLS in task_list:
                recipes.append(str(full_path).replace("multi_class_cls", "multi_label_cls"))
            if OTXTaskType.H_LABEL_CLS in task_list:
                recipes.append(str(full_path).replace("multi_class_cls", "h_label_cls"))

    return recipes


def pytest_configure(config):
    """Configure pytest options and set task, recipe, and recipe_ov lists.

    Args:
        config (pytest.Config): The pytest configuration object.

    Returns:
        None
    """
    task = config.getoption("--task")
    run_category_only = config.getoption("--run-category-only")

    # This assumes have OTX installed in environment.
    otx_module = importlib.import_module("otx")
    # Modify RECIPE_PATH based on the task
    recipe_path = Path(inspect.getfile(otx_module)).parent / "recipe"
    task_list = get_task_list(task.lower())
    recipe_dir = [find_recipe_folder(recipe_path, task_type.value.lower()) for task_type in task_list]

    # Update RECIPE_LIST
    target_recipe_list = []
    target_ov_recipe_list = []
    for task_recipe_dir in recipe_dir:
        recipe_list = [str(p) for p in task_recipe_dir.glob("**/*.yaml") if "_base_" not in p.parts]
        recipe_ov_list = [str(p) for p in task_recipe_dir.glob("**/openvino_model.yaml") if "_base_" not in p.parts]
        recipe_list = set(recipe_list) - set(recipe_ov_list)

        target_recipe_list.extend(recipe_list)
        target_ov_recipe_list.extend(recipe_ov_list)
    tile_recipe_list = [recipe for recipe in target_recipe_list if "tile" in recipe]

    # Run Model Category Recipes Only (i.e. model balance, accuracy, etc.)
    if run_category_only:
        target_recipe_list = get_model_category_list(task, recipe_path)

    pytest.TASK_LIST = task_list
    pytest.RECIPE_LIST = target_recipe_list
    pytest.RECIPE_OV_LIST = target_ov_recipe_list
    pytest.TILE_RECIPE_LIST = tile_recipe_list


@pytest.fixture(scope="session")
def fxt_asset_dir() -> Path:
    return Path(__file__).parent.parent / "assets"


# [TODO]: This is a temporary approach.
@pytest.fixture(scope="module")
def fxt_target_dataset_per_task() -> dict:
    return {
        "multi_class_cls": "tests/assets/classification_cifar10",
        "multi_class_cls_16bit": "tests/assets/classification_dataset_16bit",
        "multi_label_cls": "tests/assets/multilabel_classification_coco",
        "h_label_cls": "tests/assets/hierarchical_classification_cifar100",
        "detection": "tests/assets/detection_coco",
        "rotated_detection": "tests/assets/detection_coco",
        "instance_segmentation": "tests/assets/instance_segmentation_coco",
        "semantic_segmentation": "tests/assets/segmentation_pets",
        "keypoint_detection": "tests/assets/keypoint_detection_coco",
        "tiling_detection": "tests/assets/detection_coco",
    }
