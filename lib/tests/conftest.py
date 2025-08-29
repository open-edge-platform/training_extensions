# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest
import torch
import yaml
from datumaro import Polygon
from torch import LongTensor
from torchvision import tv_tensors
from torchvision.tv_tensors import Image, Mask

from otx.data.entity.base import ImageInfo
from otx.data.entity.torch import OTXDataBatch, OTXDataItem, OTXPredBatch, OTXPredItem
from otx.tools.converter import TEMPLATE_ID_MAPPING
from otx.types.label import HLabelInfo, LabelInfo, NullLabelInfo, SegLabelInfo
from otx.types.task import OTXTaskType
from otx.utils.device import is_xpu_available
from tests.utils import ExportCase2Test


def pytest_addoption(parser: pytest):
    """Add custom options for perf tests."""
    parser.addoption(
        "--model-category",
        action="store",
        default="all",
        choices=("speed", "balance", "accuracy", "default", "other", "all"),
        help="Choose speed|balance|accuracy|default|other|all. Defaults to all.",
    )
    parser.addoption(
        "--data-group",
        action="store",
        default="all",
        choices=("small", "medium", "large", "all"),
        help="Choose small|medium|large|all. Defaults to all.",
    )
    parser.addoption(
        "--num-repeat",
        action="store",
        default=0,
        help="Overrides default per-data-group number of repeat setting. "
        "Random seeds are set to 0 ~ num_repeat-1 for the trials. "
        "Defaults to 0 (small=3, medium=3, large=1).",
    )
    parser.addoption(
        "--num-epoch",
        action="store",
        default=0,
        help="Overrides default per-model number of epoch setting. "
        "Defaults to 0 (per-model epoch & early-stopping).",
    )
    parser.addoption(
        "--eval-upto",
        action="store",
        default="train",
        choices=("train", "export", "optimize"),
        help="Choose train|export|optimize. Defaults to train.",
    )
    parser.addoption(
        "--data-root",
        action="store",
        default="data",
        help="Dataset root directory.",
    )
    parser.addoption(
        "--output-root",
        action="store",
        help="Output root directory. Defaults to temp directory.",
    )
    parser.addoption(
        "--summary-file",
        action="store",
        help="Path to output summary file. Defaults to {output-root}/benchmark-summary.csv",
    )
    parser.addoption(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print OTX commands without execution.",
    )
    parser.addoption(
        "--deterministic",
        choices=["true", "false", "warn"],
        default=None,
        help="Turn on deterministic training (true/false/warn).",
    )
    parser.addoption(
        "--user-name",
        type=str,
        default="anonymous",
        help='Sign-off the user name who launched the regression tests this time, e.g., `--user-name "John Doe"`.',
    )
    parser.addoption(
        "--mlflow-tracking-uri",
        type=str,
        help="URI for MLFlow Tracking server to store the regression test results.",
    )
    parser.addoption(
        "--otx-ref",
        type=str,
        default="__CURRENT_BRANCH_COMMIT__",
        help="Target OTX ref (tag / branch name / commit hash) on main repo to test. Defaults to the current branch. "
        "`pip install otx[full]@https://github.com/open-edge-platform/training_extensions.git@{otx_ref}` will be executed before run, "
        "and reverted after run. Works only for v2.x assuming CLI compatibility.",
    )
    parser.addoption(
        "--resume-from",
        type=str,
        help="Previous performance test directory which contains execution results. "
        "If training was already done in previous performance test, training is skipped and refer previous result.",
    )
    parser.addoption(
        "--test-only",
        action="store",
        choices=("all", "train", "export", "optimize"),
        help="Execute test only when resume argument is given. If necessary files are not found in resume directory, "
        "necessary operations can be executed. Choose all|train|export|optimize.",
    )
    parser.addoption(
        "--open-subprocess",
        action="store_true",
        help="Open subprocess for each CLI test case. "
        "This option can be used for easy memory management "
        "while running consecutive multiple tests (default: false).",
    )
    parser.addoption(
        "--task",
        action="store",
        default="all",
        type=str,
        help="Task type of OTX to use test.",
    )
    parser.addoption(
        "--device",
        action="store",
        default="gpu",
        type=str,
        help="Which device to use.",
    )
    parser.addoption(
        "--run-category-only",
        action="store_true",
        help="Run only the model category tests that categorised as BALANCE, SPEED, ACCURACY.",
    )


@pytest.fixture(scope="session")
def fxt_multi_class_cls_data_entity() -> tuple[OTXDataItem, OTXDataBatch, OTXDataBatch]:
    img_size = (64, 64)
    fake_images = torch.zeros(size=(1, 3, *img_size), dtype=torch.float32)
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_labels = LongTensor([0])
    fake_score = torch.Tensor([0.6])
    # define data entity
    single_data_entity = OTXDataItem(image=fake_images[0], img_info=fake_image_info, label=fake_labels)
    batch_data_entity = OTXDataBatch(
        batch_size=1,
        images=fake_images,
        imgs_info=[fake_image_info],
        labels=[fake_labels],
    )
    batch_pred_data_entity = OTXPredBatch(
        batch_size=1,
        images=fake_images,
        imgs_info=[fake_image_info],
        labels=[fake_labels],
        scores=[fake_score],
    )

    return single_data_entity, batch_pred_data_entity, batch_data_entity


@pytest.fixture(scope="session")
def fxt_multi_label_cls_data_entity() -> tuple[OTXDataItem, OTXDataBatch, OTXDataBatch]:
    img_size = (64, 64)
    fake_images = torch.zeros(size=(1, 3, *img_size), dtype=torch.float32)
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_labels = LongTensor([0])
    fake_score = torch.Tensor([0.6])
    # define data entity
    single_data_entity = OTXDataItem(image=fake_images[0], img_info=fake_image_info, label=fake_labels)
    batch_data_entity = OTXDataBatch(
        batch_size=1,
        images=fake_images,
        imgs_info=[fake_image_info],
        labels=[fake_labels],
    )
    batch_pred_data_entity = OTXPredBatch(
        batch_size=1,
        images=fake_images,
        imgs_info=[fake_image_info],
        labels=[fake_labels],
        scores=[fake_score],
    )

    return single_data_entity, batch_pred_data_entity, batch_data_entity


@pytest.fixture(scope="session")
def fxt_h_label_cls_data_entity() -> tuple[OTXDataItem, OTXDataBatch, OTXPredItem]:
    img_size = (64, 64)
    fake_images = torch.zeros(size=(1, 3, *img_size), dtype=torch.float32)
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_labels = LongTensor([0])
    fake_score = torch.Tensor([0.6])
    # define data entity
    single_data_entity = OTXDataItem(image=fake_images[0], img_info=fake_image_info, label=fake_labels)
    batch_data_entity = OTXDataBatch(
        batch_size=1,
        images=fake_images,
        imgs_info=[fake_image_info],
        labels=[fake_labels],
    )
    batch_pred_data_entity = OTXPredBatch(
        batch_size=1,
        images=fake_images,
        imgs_info=[fake_image_info],
        labels=[fake_labels],
        scores=[fake_score],
    )

    return single_data_entity, batch_pred_data_entity, batch_data_entity


@pytest.fixture(scope="session")
def fxt_det_data_entity() -> tuple[tuple, OTXDataItem, OTXDataBatch]:
    img_size = (64, 64)
    fake_image = torch.zeros(size=(3, *img_size), dtype=torch.float32)
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_bboxes = tv_tensors.BoundingBoxes(data=torch.Tensor([0, 0, 5, 5]), format="xyxy", canvas_size=(10, 10))
    fake_labels = LongTensor([1])
    # define data entity
    single_data_entity = OTXDataItem(
        image=fake_image,
        img_info=fake_image_info,
        bboxes=fake_bboxes,
        label=fake_labels,
    )
    batch_data_entity = OTXDataBatch(
        batch_size=1,
        images=[Image(fake_image)],
        imgs_info=[fake_image_info],
        bboxes=[fake_bboxes],
        labels=[fake_labels],
    )
    batch_pred_data_entity = OTXPredBatch(
        batch_size=1,
        images=[Image(fake_image)],
        imgs_info=[fake_image_info],
        bboxes=[fake_bboxes],
        labels=[fake_labels],
        scores=[],
    )

    return single_data_entity, batch_pred_data_entity, batch_data_entity


@pytest.fixture(scope="session")
def fxt_inst_seg_data_entity() -> tuple[tuple, OTXDataItem, OTXDataBatch]:
    img_size = (64, 64)
    fake_image = torch.zeros(size=(3, *img_size), dtype=torch.float32)
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_bboxes = tv_tensors.BoundingBoxes(data=torch.Tensor([0, 0, 5, 5]), format="xyxy", canvas_size=(10, 10))
    fake_labels = LongTensor([1])
    fake_masks = Mask(torch.randint(low=0, high=255, size=(1, *img_size), dtype=torch.uint8))
    fake_polygons = [Polygon(points=[1, 1, 2, 2, 3, 3, 4, 4])]
    # define data entity
    single_data_entity = OTXDataItem(
        image=fake_image,
        img_info=fake_image_info,
        bboxes=fake_bboxes,
        masks=fake_masks,
        label=fake_labels,
        polygons=fake_polygons,
    )
    batch_data_entity = OTXDataBatch(
        batch_size=1,
        images=[Image(data=fake_image)],
        imgs_info=[fake_image_info],
        bboxes=[fake_bboxes],
        labels=[fake_labels],
        masks=[fake_masks],
        polygons=[fake_polygons],
    )
    batch_pred_data_entity = OTXPredBatch(
        batch_size=1,
        images=[Image(data=fake_image)],
        imgs_info=[fake_image_info],
        bboxes=[fake_bboxes],
        labels=[fake_labels],
        masks=[fake_masks],
        polygons=[fake_polygons],
    )

    return single_data_entity, batch_pred_data_entity, batch_data_entity


@pytest.fixture(scope="session")
def fxt_seg_data_entity() -> tuple[tuple, OTXDataItem, OTXDataBatch]:
    img_size = (32, 32)
    fake_image = torch.zeros(size=(3, *img_size), dtype=torch.uint8).numpy()
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_masks = Mask(torch.randint(low=0, high=2, size=img_size, dtype=torch.uint8))
    # define data entity
    single_data_entity = OTXDataItem(
        image=fake_image,
        img_info=fake_image_info,
        masks=fake_masks,
    )
    batch_data_entity = OTXDataBatch(
        batch_size=1,
        images=[Image(data=torch.from_numpy(fake_image))],
        imgs_info=[fake_image_info],
        masks=[fake_masks],
    )
    batch_pred_data_entity = OTXPredItem(
        batch_size=1,
        images=[Image(data=torch.from_numpy(fake_image))],
        imgs_info=[fake_image_info],
        masks=[fake_masks],
        scores=[],
    )

    return single_data_entity, batch_pred_data_entity, batch_data_entity


@pytest.fixture(scope="session")
def fxt_accelerator(request: pytest.FixtureRequest) -> str:
    if is_xpu_available():
        return "xpu"
    return request.config.getoption("--device", "gpu")


@pytest.fixture(params=set(OTXTaskType))
def fxt_task(request: pytest.FixtureRequest) -> OTXTaskType:
    return request.param


@pytest.fixture(scope="session", autouse=True)
def fxt_null_label_info() -> LabelInfo:
    return NullLabelInfo()


@pytest.fixture(scope="session", autouse=True)
def fxt_seg_label_info() -> SegLabelInfo:
    label_names = ["class1", "class2", "class3"]
    return SegLabelInfo(
        label_names=label_names,
        label_groups=[
            label_names,
            ["class2", "class3"],
        ],
        label_ids=["0", "1", "2"],
    )


@pytest.fixture(scope="session", autouse=True)
def fxt_multiclass_labelinfo() -> LabelInfo:
    label_names = ["class1", "class2", "class3"]
    return LabelInfo(
        label_names=label_names,
        label_groups=[
            label_names,
            ["class2", "class3"],
        ],
        label_ids=["0", "1", "2"],
    )


@pytest.fixture(scope="session", autouse=True)
def fxt_multilabel_labelinfo() -> LabelInfo:
    label_names = ["class1", "class2", "class3"]
    return LabelInfo(
        label_names=label_names,
        label_groups=[
            [label_names[0]],
            [label_names[1]],
            [label_names[2]],
        ],
        label_ids=["0", "1", "2"],
    )


@pytest.fixture()
def fxt_hlabel_multilabel_info() -> HLabelInfo:
    return HLabelInfo(
        label_names=[
            "Heart",
            "Spade",
            "Heart_Queen",
            "Heart_King",
            "Spade_A",
            "Spade_King",
            "Black_Joker",
            "Red_Joker",
            "Extra_Joker",
        ],
        label_groups=[
            ["Heart", "Spade"],
            ["Heart_Queen", "Heart_King"],
            ["Spade_A", "Spade_King"],
            ["Black_Joker"],
            ["Red_Joker"],
            ["Extra_Joker"],
        ],
        num_multiclass_heads=3,
        num_multilabel_classes=3,
        head_idx_to_logits_range={"0": (0, 2), "1": (2, 4), "2": (4, 6)},
        num_single_label_classes=3,
        empty_multiclass_head_indices=[],
        class_to_group_idx={
            "Heart": (0, 0),
            "Spade": (0, 1),
            "Heart_Queen": (1, 0),
            "Heart_King": (1, 1),
            "Spade_A": (2, 0),
            "Spade_King": (2, 1),
            "Black_Joker": (3, 0),
            "Red_Joker": (3, 1),
            "Extra_Joker": (3, 2),
        },
        all_groups=[
            ["Heart", "Spade"],
            ["Heart_Queen", "Heart_King"],
            ["Spade_A", "Spade_King"],
            ["Black_Joker"],
            ["Red_Joker"],
            ["Extra_Joker"],
        ],
        label_to_idx={
            "Heart": 0,
            "Spade": 1,
            "Heart_Queen": 2,
            "Heart_King": 3,
            "Spade_A": 4,
            "Spade_King": 5,
            "Black_Joker": 6,
            "Red_Joker": 7,
            "Extra_Joker": 8,
        },
        label_tree_edges=[
            ["Heart_Queen", "Heart"],
            ["Heart_King", "Heart"],
            ["Spade_A", "Spade"],
            ["Spade_King", "Spade"],
        ],
        label_ids=[str(i) for i in range(9)],
    )


@pytest.fixture()
def fxt_export_list() -> list[ExportCase2Test]:
    return [
        ExportCase2Test("ONNX", False, "exported_model.onnx"),
        ExportCase2Test("OPENVINO", False, "exported_model.xml"),
        ExportCase2Test("OPENVINO", True, "exportable_code.zip"),
    ]


def get_model_template_paths() -> dict[OTXTaskType, list[dict]]:
    """Get Geti model template paths from the templates directory.

    Returns:
        dict: A dictionary mapping task types to lists of template paths and tiling options.
    """

    from otx.backend.native.cli.utils import get_otx_root_path

    template_dir = Path(get_otx_root_path()).parent.parent / "tests" / "assets" / "geti" / "model_configs"
    template_paths = template_dir.rglob("*.yaml")
    template_dict = defaultdict(list)

    for template_path in template_paths:
        with template_path.open() as file:
            template = yaml.safe_load(file)

        model_id = template.get("model_manifest_id")

        model_config_path = TEMPLATE_ID_MAPPING[model_id]["recipe_path"]
        model_task = OTXTaskType(model_config_path.parent.name.upper())
        has_tiling = (
            template["hyperparameters"].get("dataset_preparation", {}).get("augmentation", {}).get("tiling", None)
        )

        # Add base (no-tiling)
        template_dict[model_task].append(
            {
                "template_path": template_path,
                "tiling": False,
            },
        )

        # Add tiling version if available
        if has_tiling:
            template_dict[model_task].append(
                {
                    "template_path": template_path,
                    "tiling": True,
                },
            )

    # Alias multi-class template for multi-label and hierarchical
    if OTXTaskType.MULTI_CLASS_CLS in template_dict:
        template_dict[OTXTaskType.MULTI_LABEL_CLS] = template_dict[OTXTaskType.MULTI_CLASS_CLS]
        template_dict[OTXTaskType.H_LABEL_CLS] = template_dict[OTXTaskType.MULTI_CLASS_CLS]

    return template_dict


def pytest_generate_tests(metafunc):
    """
    Dynamically generates parameterized test cases for each available task template.

    If the test function requires the 'task_template' fixture, this hook loads model templates
    based on the specified --task and --run-category-only command-line options. It then creates
    combinations of (task_enum, template_path, tiling_flag) and registers them as individual
    test cases using pytest's parametrize mechanism, with readable test IDs for clarity.
    """
    if "task_template" in metafunc.fixturenames:
        task_name = metafunc.config.getoption("task")
        template_dict = get_model_template_paths()

        params = []
        if task_name.lower() == "all":
            params = [
                (task, entry["template_path"], entry["tiling"])
                for task, entries in template_dict.items()
                for entry in entries
            ]
        else:
            task_enum = OTXTaskType(task_name.upper())
            params = [
                (task_enum, entry["template_path"], entry["tiling"]) for entry in template_dict.get(task_enum, [])
            ]

        ids = [f"{task.name}/{path.parent.name}" + ("/tiling" if tiling else "") for task, path, tiling in params]

        metafunc.parametrize("task_template", params, ids=ids)
