# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download a ~40-sample CIFAR-100 subset and export it as a hierarchical classification dataset.

CIFAR-100 has 100 fine-grained classes grouped into 20 superclasses, making it
ideal for hierarchical classification testing.  We use 10 superclasses to keep
the dataset small.

Usage
-----
    python tests/assets/hierarchical_classification_cifar100/download.py [--output_dir OUTPUT_DIR]
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datumaro.experimental import Dataset
from datumaro.experimental.categories import (
    HierarchicalLabelCategories,
    HierarchicalLabelCategory,
    LabelGroup,
)
from datumaro.experimental.export_import import export_dataset
from datumaro.experimental.fields import ImageInfo as DmImageInfo
from datumaro.experimental.fields import Subset
from torchvision import tv_tensors
from torchvision.datasets import CIFAR100

from getitune.data.entity.sample import ClassificationHierarchicalSample

# CIFAR-100 superclass (coarse) names
CIFAR100_COARSE_LABELS: tuple[str, ...] = (
    "aquatic_mammals",
    "fish",
    "flowers",
    "food_containers",
    "fruit_and_vegetables",
    "household_electrical_devices",
    "household_furniture",
    "insects",
    "large_carnivores",
    "large_man-made_outdoor_things",
    "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores",
    "medium_mammals",
    "non-insect_invertebrates",
    "people",
    "reptiles",
    "small_mammals",
    "trees",
    "vehicles_1",
    "vehicles_2",
)

# CIFAR-100 fine-grained class names (official ordering)
CIFAR100_FINE_LABELS: tuple[str, ...] = (
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
)

# Mapping from fine label index to coarse label index
FINE_TO_COARSE: tuple[int, ...] = (
    4,
    1,
    14,
    8,
    0,
    6,
    7,
    7,
    18,
    3,
    3,
    14,
    9,
    18,
    7,
    11,
    3,
    9,
    7,
    11,
    6,
    14,
    5,
    10,
    7,
    6,
    13,
    15,
    3,
    15,
    0,
    11,
    1,
    10,
    12,
    14,
    12,
    6,
    11,
    5,
    5,
    19,
    8,
    8,
    15,
    13,
    14,
    17,
    18,
    10,
    16,
    4,
    17,
    4,
    4,
    0,
    17,
    4,
    18,
    17,
    10,
    3,
    4,
    16,
    12,
    12,
    16,
    1,
    10,
    19,
    4,
    10,
    0,
    1,
    16,
    16,
    9,
    13,
    15,
    13,
    16,
    19,
    4,
    4,
    6,
    19,
    5,
    5,
    8,
    19,
    18,
    1,
    4,
    15,
    6,
    0,
    17,
    8,
    14,
    7,
)

SAMPLES_PER_CLASS_TRAIN: int = 2
SAMPLES_PER_CLASS_VAL: int = 1
SAMPLES_PER_CLASS_TEST: int = 1
NUM_COARSE_CLASSES: int = 10
# Indices of the 10 superclasses we keep (of 20 total)
SELECTED_COARSE_INDICES: frozenset[int] = frozenset({0, 1, 4, 7, 8, 10, 11, 14, 15, 18})
# Target: ~40 samples (10 coarse classes x 4 per class, 2/1/1 split)


def _select_samples(
    cifar_dataset: CIFAR100,
    samples_per_class: int,
) -> list[tuple[np.ndarray, int, int, int]]:
    """Select a balanced subset from a CIFAR-100 split (balanced by coarse label).

    Only superclasses in ``SELECTED_COARSE_INDICES`` are kept.

    Returns a list of (image_HWC, fine_label, coarse_label, global_idx) tuples.
    """
    class_counts: dict[int, int] = defaultdict(int)
    selected: list[tuple[np.ndarray, int, int, int]] = []

    for idx in range(len(cifar_dataset)):
        img, fine_label = cifar_dataset[idx]
        coarse_label = FINE_TO_COARSE[fine_label]
        if coarse_label not in SELECTED_COARSE_INDICES:
            continue
        if class_counts[coarse_label] >= samples_per_class:
            continue
        img_np: np.ndarray = np.array(img, dtype=np.uint8)
        selected.append((img_np, fine_label, coarse_label, idx))
        class_counts[coarse_label] += 1
        if all(c >= samples_per_class for c in class_counts.values()) and len(class_counts) >= NUM_COARSE_CLASSES:
            break

    return selected


def _build_dataset(
    train_items: list[tuple[np.ndarray, int, int, int]],
    val_items: list[tuple[np.ndarray, int, int, int]],
    test_items: list[tuple[np.ndarray, int, int, int]],
    images_dir: Path,
) -> Dataset:
    """Build a ``datumaro.experimental.Dataset`` of ``ClassificationHierarchicalSample``."""
    from PIL import Image as PILImage

    # Collect only the coarse/fine labels that actually appear in the data.
    all_items = train_items + val_items + test_items
    used_coarse: set[int] = {coarse for _, _, coarse, _ in all_items}
    used_fine: set[int] = {fine for _, fine, _, _ in all_items}

    coarse_sorted = sorted(used_coarse)
    fine_sorted = sorted(used_fine)
    # Items contain both coarse (root) and fine labels.
    # HierarchicalLabelCategories requires that parent names exist as items.
    # Coarse labels are listed first (as roots), then fine labels.
    num_used_coarse = len(coarse_sorted)
    fine_remap = {old: num_used_coarse + new for new, old in enumerate(fine_sorted)}

    # Build HierarchicalLabelCategory items:
    # - coarse labels as roots (no parent)
    # - fine labels pointing to their coarse superclass
    hlabel_items: list[HierarchicalLabelCategory] = []
    for coarse_idx in coarse_sorted:
        hlabel_items.append(HierarchicalLabelCategory(name=CIFAR100_COARSE_LABELS[coarse_idx], parent=""))  # noqa: PERF401
    for fine_idx in fine_sorted:
        fine_name = CIFAR100_FINE_LABELS[fine_idx]
        coarse_idx = FINE_TO_COARSE[fine_idx]
        parent_name = CIFAR100_COARSE_LABELS[coarse_idx]
        hlabel_items.append(HierarchicalLabelCategory(name=fine_name, parent=parent_name))

    # Build label groups: one exclusive group per coarse class containing its fine children.
    label_groups: list[LabelGroup] = []
    coarse_to_fine: dict[int, list[str]] = defaultdict(list)
    for fine_idx in fine_sorted:
        coarse_idx = FINE_TO_COARSE[fine_idx]
        coarse_to_fine[coarse_idx].append(CIFAR100_FINE_LABELS[fine_idx])

    for coarse_idx in coarse_sorted:
        coarse_name = CIFAR100_COARSE_LABELS[coarse_idx]
        children = coarse_to_fine.get(coarse_idx, [])
        label_groups.append(LabelGroup(name=coarse_name, labels=tuple(children)))

    categories = {
        "label": HierarchicalLabelCategories(
            items=tuple(hlabel_items),
            label_groups=tuple(label_groups),
        ),
    }
    dataset: Dataset = Dataset(ClassificationHierarchicalSample, categories=categories)  # type: ignore[arg-type]

    images_dir.mkdir(parents=True, exist_ok=True)

    def _add_items(
        items: list[tuple[np.ndarray, int, int, int]],
        subset: Subset,
        prefix: str,
    ) -> None:
        for img_np, fine_label, _coarse_label, global_idx in items:
            h, w = img_np.shape[:2]
            img_chw = torch.from_numpy(img_np.transpose(2, 0, 1))
            image = tv_tensors.Image(img_chw)

            filename = f"{prefix}_{global_idx:05d}.png"
            PILImage.fromarray(img_np).save(images_dir / filename)

            # Hierarchical label: only fine label index (hierarchy is encoded in categories)
            label = torch.tensor([fine_remap[fine_label]], dtype=torch.long)

            sample = ClassificationHierarchicalSample(
                image=image,
                label=label,
                dm_image_info=DmImageInfo(width=w, height=h),
                subset=subset,
            )
            dataset.append(sample)

    _add_items(train_items, Subset.TRAINING, "train")
    _add_items(val_items, Subset.VALIDATION, "val")
    _add_items(test_items, Subset.TESTING, "test")

    return dataset


def main(output_dir: Path | None = None) -> None:
    """Download CIFAR-100, build a 40-sample hierarchical classification dataset, and export it."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent

    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir = output_dir / "raw"
    images_dir = output_dir / "images"

    print(f"Downloading CIFAR-100 to {download_dir} ...")
    train_set = CIFAR100(root=str(download_dir), train=True, download=True)
    test_set = CIFAR100(root=str(download_dir), train=False, download=True)

    print("Selecting balanced subset ...")
    train_items = _select_samples(train_set, SAMPLES_PER_CLASS_TRAIN)
    valtest_items = _select_samples(test_set, SAMPLES_PER_CLASS_VAL + SAMPLES_PER_CLASS_TEST)
    mid = len(valtest_items) // 2
    val_items = valtest_items[:mid]
    test_items = valtest_items[mid:]
    print(f"  Training   samples: {len(train_items)}")
    print(f"  Validation samples: {len(val_items)}")
    print(f"  Testing    samples: {len(test_items)}")
    print(f"  Total:              {len(train_items) + len(val_items) + len(test_items)}")

    print("Building datumaro experimental dataset with ClassificationHierarchicalSample ...")
    dataset = _build_dataset(train_items, val_items, test_items, images_dir)
    print(f"  Dataset length: {len(dataset)}")

    print(f"Exporting dataset to {output_dir} ...")
    export_dataset(dataset, output_dir)

    print("Cleaning up intermediate files ...")
    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(images_dir, ignore_errors=True)

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a 40-sample CIFAR-100 hierarchical classification benchmark dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=("Directory to save the dataset. Defaults to tests/assets/hierarchical_classification_cifar100"),
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir)
