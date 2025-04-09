# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest
from datumaro import Label
from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.media import Image

from otx.core.config import register_configs


@pytest.fixture(scope="session", autouse=True)
def fxt_register_configs() -> None:
    register_configs()


@pytest.fixture()
def fxt_hlabel_dataset_subset() -> Dataset:
    return Dataset.from_iterable(
        [
            DatasetItem(
                id=0,
                subset="train",
                media=Image.from_numpy(np.zeros((3, 10, 10))),
                annotations=[
                    Label(
                        label=2,
                        id=0,
                        group=1,
                    ),
                ],
            ),
            DatasetItem(
                id=1,
                subset="train",
                media=Image.from_numpy(np.zeros((3, 10, 10))),
                annotations=[
                    Label(
                        label=4,
                        id=0,
                        group=2,
                    ),
                ],
            ),
        ],
        categories={
            AnnotationType.label: LabelCategories(
                items=[
                    LabelCategories.Category(name="Heart", parent=""),
                    LabelCategories.Category(name="Spade", parent=""),
                    LabelCategories.Category(name="Heart_Queen", parent="Heart"),
                    LabelCategories.Category(name="Heart_King", parent="Heart"),
                    LabelCategories.Category(name="Spade_A", parent="Spade"),
                    LabelCategories.Category(name="Spade_King", parent="Spade"),
                    LabelCategories.Category(name="Black_Joker", parent=""),
                    LabelCategories.Category(name="Red_Joker", parent=""),
                    LabelCategories.Category(name="Extra_Joker", parent=""),
                ],
                label_groups=[
                    LabelCategories.LabelGroup(name="Card", labels=["Heart", "Spade"]),
                    LabelCategories.LabelGroup(name="Heart Group", labels=["Heart_Queen", "Heart_King"]),
                    LabelCategories.LabelGroup(name="Spade Group", labels=["Spade_Queen", "Spade_King"]),
                ],
            ),
        },
    ).get_subset("train")
