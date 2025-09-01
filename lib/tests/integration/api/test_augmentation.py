# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools

import pytest
from datumaro import Dataset as DmDataset

from otx.config.data import SamplerConfig, SubsetConfig
from otx.data.factory import OTXDatasetFactory
from otx.tools.auto_configurator import AutoConfigurator
from otx.types.task import OTXTaskType


def _test_augmentation(
    recipe: str,
    target_dataset_per_task: dict,
    configurable_augs: list[str],
) -> None:
    # Load recipe
    recipe_tokens = recipe.split("/")
    task_name = recipe_tokens[-2]
    task = OTXTaskType(task_name.upper())
    config = AutoConfigurator(
        data_root=target_dataset_per_task[task_name],
        task=task,
        model_config_path=recipe,
    ).config
    train_config = config["data"]["train_subset"]
    train_config["input_size"] = (32, 32)
    data_format = config["data"]["data_format"]

    # Load dataset
    dm_dataset = DmDataset.import_from(
        target_dataset_per_task[task_name],
        format=data_format,
    )

    # Evaluate all on/off aug combinations
    img_shape = None
    for switches in itertools.product([True, False], repeat=len(configurable_augs)):
        # Configure on/off
        for aug_name, switch in zip(configurable_augs, switches):
            aug_found = False
            for aug_config in train_config["transforms"]:
                if aug_name in aug_config["class_path"]:
                    aug_config["enable"] = switch
                    aug_found = True
                    break
            assert aug_found, f"{aug_name} not found in {recipe}"
        # Create dataset
        dataset = OTXDatasetFactory.create(
            task=task,
            dm_subset=dm_dataset,
            cfg_subset=SubsetConfig(sampler=SamplerConfig(**train_config.pop("sampler", {})), **train_config),
            data_format=data_format,
        )
        # Check if all aug combinations are size-compatible
        data = dataset[0]
        if not img_shape:
            img_shape = data.img_info.img_shape
        else:
            assert img_shape == data.img_info.img_shape


CLS_RECIPES = [recipe for recipe in pytest.RECIPE_LIST if "_cls" in recipe and "tv_" not in recipe]
DET_RECIPES = [recipe for recipe in pytest.RECIPE_LIST if "/detection/" in recipe and "deim" not in recipe]
INST_SEG_RECIPES = [recipe for recipe in pytest.RECIPE_LIST if "/instance_segmentation/" in recipe]
SEM_SEG_RECIPES = [recipe for recipe in pytest.RECIPE_LIST if "/semantic_segmentation/" in recipe]
KP_DET_RECIPES = [recipe for recipe in pytest.RECIPE_LIST if "/keypoint_detection/" in recipe]


@pytest.mark.parametrize("recipe", CLS_RECIPES + DET_RECIPES + INST_SEG_RECIPES)
def test_augmentation(
    recipe: str,
    fxt_target_dataset_per_task: dict,
):
    configurable_augs = [
        "RandomPhotometricDistort",
        "RandomAffine",
        "RandomVerticalFlip",
        "RandomGaussianBlur",
        "RandomFlip",
        "RandomGaussianNoise",
    ]
    _test_augmentation(recipe, fxt_target_dataset_per_task, configurable_augs)


@pytest.mark.parametrize("recipe", SEM_SEG_RECIPES)
def test_augmentation_seg(
    recipe: str,
    fxt_target_dataset_per_task: dict,
):
    configurable_augs = [
        "PhotoMetricDistortion",
        "RandomAffine",
        "RandomVerticalFlip",
        "RandomGaussianBlur",
        "RandomFlip",
        "RandomGaussianNoise",
    ]
    _test_augmentation(recipe, fxt_target_dataset_per_task, configurable_augs)


@pytest.mark.parametrize("recipe", KP_DET_RECIPES)
def test_augmentation_kp_det(
    recipe: str,
    fxt_target_dataset_per_task: dict,
):
    configurable_augs = [
        "TopdownAffine",
        "RandomPhotometricDistort",
        "RandomGaussianBlur",
        "RandomGaussianNoise",
    ]
    _test_augmentation(recipe, fxt_target_dataset_per_task, configurable_augs)
    _test_augmentation(recipe, fxt_target_dataset_per_task, configurable_augs)
