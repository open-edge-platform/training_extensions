# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import importlib
import inspect
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

from otx.core.types.task import OTXTaskType
from otx.tools.converter import TEMPLATE_ID_DICT


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
    elif task == "visual_prompting":
        tasks = [OTXTaskType.VISUAL_PROMPTING]
    elif task == "anomaly":
        tasks = [OTXTaskType.ANOMALY]
    elif task == "keypoint_detection":
        tasks = [OTXTaskType.KEYPOINT_DETECTION]
    else:
        tasks = [OTXTaskType(task.upper())]
    return tasks


def get_model_category_list(task) -> list[str]:
    """
    Retrieve the list of model categories from `otx/tools/templates`.

    This function extracts `model_category` values from `template.yaml`, which may include
    categories such as "balance" or "accuracy." It then maps each category to its corresponding
    recipe in `otx/recipe`.

    Args:
        task (str): The task for which to retrieve model categories.

    Raises:
        FileNotFoundError: If no recipe is found for the specified task.

    Returns:
        list[str]: A list of recipe paths.
    """

    # Locate the OTX module and relevant directories
    otx_module = importlib.import_module("otx")
    otx_root = Path(inspect.getfile(otx_module)).parent
    template_dir = otx_root / "tools" / "templates"
    recipe_dir = otx_root / "recipe"

    # Collect all template.yaml files
    template_paths = template_dir.rglob("template.yaml")
    template_dict = defaultdict(list)

    # Extract model categories from templates
    for template_path in template_paths:
        with Path.open(template_path) as file:
            template = yaml.safe_load(file)

        if "model_category" in template and "model_template_id" in template:
            model_id = template["model_template_id"]
            model_name = TEMPLATE_ID_DICT[model_id]["model_name"]
            model_task = TEMPLATE_ID_DICT[model_id]["task"]
            template_dict[model_task].append(model_name)

    # Extend classification categories
    template_dict[OTXTaskType.MULTI_LABEL_CLS] = template_dict[OTXTaskType.MULTI_CLASS_CLS]
    template_dict[OTXTaskType.H_LABEL_CLS] = template_dict[OTXTaskType.MULTI_CLASS_CLS]

    # Get the list of tasks to search for recipes
    task_list = get_task_list(task.lower())
    recipes = set()

    # Find matching recipes for each model
    for task_type in task_list:
        if task_type not in template_dict:
            continue

        task_recipe_dirs = list(recipe_dir.rglob(task_type.lower()))
        if not task_recipe_dirs:
            continue

        task_recipe_dir = task_recipe_dirs[0]  # Assuming one directory per task

        for model_name in template_dict[task_type]:
            recipe_paths = task_recipe_dir.rglob(f"{model_name}*.yaml")
            recipes.update(recipe_paths)

    if not recipes:
        msg = f"No recipe found for task '{task}'."
        raise FileNotFoundError(msg)

    return [str(recipe) for recipe in recipes]


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
        target_recipe_list = get_model_category_list(task)

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
        "multi_class_cls": "tests/assets/classification_dataset",
        "multi_label_cls": "tests/assets/multilabel_classification",
        "h_label_cls": "tests/assets/hlabel_classification",
        "detection": "tests/assets/car_tree_bug",
        "rotated_detection": "tests/assets/car_tree_bug",
        "instance_segmentation": "tests/assets/car_tree_bug",
        "semantic_segmentation": "tests/assets/common_semantic_segmentation_dataset",
        "visual_prompting": "tests/assets/car_tree_bug",
        "anomaly": "tests/assets/anomaly_hazelnut",
        "keypoint_detection": "tests/assets/car_tree_bug_keypoint",
        "tiling_detection": "tests/assets/tiling_small_objects",
    }


@pytest.fixture(scope="module")
def fxt_cli_override_command_per_task() -> dict:
    return {
        "multi_class_cls": [],
        "multi_label_cls": [],
        "h_label_cls": [],
        "detection": [],
        "rotated_detection": [],
        "instance_segmentation": [],
        "semantic_segmentation": [],
        "visual_prompting": [],
        "anomaly": [],
        "keypoint_detection": [],
    }
