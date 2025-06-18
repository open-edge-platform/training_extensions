# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from datumaro.components.annotation import AnnotationType, Bbox, Label, LabelCategories, Mask, Polygon
from datumaro.components.dataset import Dataset as DmDataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image

from otx.data.dataset.anomaly import OTXAnomalyDataset
from otx.data.dataset.classification import (
    HLabelInfo,
    OTXHlabelClsDataset,
    OTXMulticlassClsDataset,
    OTXMultilabelClsDataset,
)
from otx.data.dataset.detection import (
    OTXDetectionDataset,
)
from otx.data.dataset.instance_segmentation import OTXInstanceSegDataset
from otx.data.dataset.segmentation import (
    OTXSegmentationDataset,
)
from otx.data.entity.torch import OTXDataItem
from otx.types.task import OTXTaskType

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from otx.data.dataset.base import OTXDataset

_LABEL_NAMES = ["Non-Rigid", "Rigid", "Rectangle", "Triangle", "Circle", "Lion", "Panda"]


@pytest.fixture(params=["bytes", "file"])
def fxt_dm_item(request, tmpdir) -> DatasetItem:
    np_img = np.zeros(shape=(10, 10, 3), dtype=np.uint8)
    np_img[:, :, 0] = 0  # Set 0 for B channel
    np_img[:, :, 1] = 1  # Set 1 for G channel
    np_img[:, :, 2] = 2  # Set 2 for R channel

    if request.param == "bytes":
        _, np_bytes = cv2.imencode(".png", np_img)
        media = Image.from_bytes(np_bytes.tobytes())
        media.path = ""
    elif request.param == "file":
        fname = str(uuid.uuid4())
        fpath = str(Path(tmpdir) / f"{fname}.png")
        cv2.imwrite(fpath, np_img)
        media = Image.from_file(fpath)
    else:
        raise ValueError(request.param)

    return DatasetItem(
        id="item",
        subset="train",
        media=media,
        annotations=[
            Label(label=0),
            Bbox(x=200, y=200, w=1, h=1, label=0),
            Mask(label=0, image=np.eye(10, dtype=np.uint8)),
            Polygon(points=[399.0, 570.0, 397.0, 572.0, 397.0, 573.0, 394.0, 576.0], label=0),
        ],
    )


@pytest.fixture(params=["bytes", "file"])
def fxt_dm_item_bbox_only(request, tmpdir) -> DatasetItem:
    np_img = np.zeros(shape=(10, 10, 3), dtype=np.uint8)
    np_img[:, :, 0] = 0  # Set 0 for B channel
    np_img[:, :, 1] = 1  # Set 1 for G channel
    np_img[:, :, 2] = 2  # Set 2 for R channel

    if request.param == "bytes":
        _, np_bytes = cv2.imencode(".png", np_img)
        media = Image.from_bytes(np_bytes.tobytes())
        media.path = ""
    elif request.param == "file":
        fname = str(uuid.uuid4())
        fpath = str(Path(tmpdir) / f"{fname}.png")
        cv2.imwrite(fpath, np_img)
        media = Image.from_file(fpath)
    else:
        raise ValueError(request.param)

    return DatasetItem(
        id="item",
        subset="train",
        media=media,
        annotations=[
            Bbox(x=0, y=0, w=1, h=1, label=0),
            Bbox(x=1, y=0, w=1, h=1, label=0),
            Bbox(x=1, y=1, w=1, h=1, label=0),
        ],
    )


@pytest.fixture()
def fxt_mock_dm_subset(mocker: MockerFixture, fxt_dm_item: DatasetItem) -> MagicMock:
    mock_dm_subset = mocker.MagicMock(spec=DmDataset)
    mock_dm_subset.__getitem__.return_value = fxt_dm_item
    mock_dm_subset.__len__.return_value = 1
    mock_dm_subset.categories().__getitem__.return_value = LabelCategories.from_iterable(_LABEL_NAMES)
    mock_dm_subset.ann_types.return_value = [
        AnnotationType.label,
        AnnotationType.bbox,
        AnnotationType.mask,
        AnnotationType.polygon,
    ]
    return mock_dm_subset


@pytest.fixture()
def fxt_mock_det_dm_subset(mocker: MockerFixture, fxt_dm_item_bbox_only: DatasetItem) -> MagicMock:
    mock_dm_subset = mocker.MagicMock(spec=DmDataset)
    mock_dm_subset.__getitem__.return_value = fxt_dm_item_bbox_only
    mock_dm_subset.__len__.return_value = 1
    mock_dm_subset.categories().__getitem__.return_value = LabelCategories.from_iterable(_LABEL_NAMES)
    mock_dm_subset.ann_types.return_value = [AnnotationType.bbox]
    return mock_dm_subset


@pytest.fixture(
    params=[
        (OTXHlabelClsDataset, OTXDataItem, {}),
        (OTXMultilabelClsDataset, OTXDataItem, {}),
        (OTXMulticlassClsDataset, OTXDataItem, {}),
        (OTXDetectionDataset, OTXDataItem, {}),
        (OTXInstanceSegDataset, OTXDataItem, {"include_polygons": True}),
        (OTXSegmentationDataset, OTXDataItem, {}),
        (OTXAnomalyDataset, OTXDataItem, {"task_type": OTXTaskType.ANOMALY}),
        (OTXAnomalyDataset, OTXDataItem, {"task_type": OTXTaskType.ANOMALY_CLASSIFICATION}),
        (OTXAnomalyDataset, OTXDataItem, {"task_type": OTXTaskType.ANOMALY_DETECTION}),
        (OTXAnomalyDataset, OTXDataItem, {"task_type": OTXTaskType.ANOMALY_SEGMENTATION}),
    ],
    ids=[
        "hlabel_cls",
        "multi_label_cls",
        "multi_class_cls",
        "detection",
        "instance_seg",
        "semantic_seg",
        "anomaly",
        "anomaly_cls",
        "anomaly_det",
        "anomaly_seg",
    ],
)
def fxt_dataset_and_data_entity_cls(
    request: pytest.FixtureRequest,
) -> tuple[OTXDataset, OTXDataItem]:
    return request.param


@pytest.fixture()
def fxt_mock_hlabelinfo():
    mock_dict = MagicMock()
    mock_dict.__getitem__.return_value = (0, 0)
    return HLabelInfo(
        label_names=_LABEL_NAMES,
        label_groups=[["Non-Rigid", "Rigid"], ["Rectangle", "Triangle"], ["Circle"], ["Lion"], ["Panda"]],
        label_ids=_LABEL_NAMES,
        num_multiclass_heads=2,
        num_multilabel_classes=3,
        head_idx_to_logits_range={"0": (0, 2), "1": (2, 4)},
        num_single_label_classes=4,
        class_to_group_idx=mock_dict,
        all_groups=[["Non-Rigid", "Rigid"], ["Rectangle", "Triangle"], ["Circle"], ["Lion"], ["Panda"]],
        label_to_idx={
            "Rigid": 0,
            "Rectangle": 1,
            "Triangle": 2,
            "Non-Rigid": 3,
            "Circle": 4,
            "Lion": 5,
            "Panda": 6,
        },
        label_tree_edges=[
            ["Rectangle", "Rigid"],
            ["Triangle", "Rigid"],
            ["Circle", "Non-Rigid"],
        ],
        empty_multiclass_head_indices=[],
    )


@pytest.fixture()
def fxt_hlabel_dataset_subset() -> DmDataset:
    return DmDataset.from_iterable(
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
