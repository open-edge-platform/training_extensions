# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared helper for Roboflow-100 HuggingFace detection mirrors.

The Roboflow-100 (RF100) benchmark datasets republished under the
``Francesco/*`` namespace on the HuggingFace Hub all share a single parquet
schema::

    image_id : int64
    image    : struct<bytes: binary, path: string>   # embedded JPEG/PNG
    width    : int32
    height   : int32
    objects  : struct<
        id       : list<int64>,
        area     : list<int64>,
        bbox     : list<fixed_size_list<float>[4]>,   # COCO [x, y, w, h]
        category : list<int64>,                        # ClassLabel index
    >

Every dataset ships native ``train`` / ``validation`` / ``test`` parquet splits,
so no synthetic split needs to be carved out. This module downloads those
splits and converts them into the experimental Datumaro ``CocoSample`` format
used by the detection benchmarks.

Because the schema is identical across all RF100 mirrors, the individual
``scripts/benchmark_datasets/prepare_*.py`` scripts only need to declare the
repository, a pinned revision, the label names, and the per-split parquet file
names — the conversion logic lives here.
"""

from __future__ import annotations

import logging
import shutil
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq
from datumaro.experimental import Dataset, LazyImage
from datumaro.experimental.data_formats.coco.sample import CocoCategories, CocoSample
from datumaro.experimental.export_import import export_dataset
from datumaro.experimental.fields import ImageInfo, Subset

from getitune.benchmark.dataset_helpers import download

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from getitune.benchmark.dataset_helpers import DatasetArgs

logger = logging.getLogger(__name__)

# Map upstream HuggingFace split names to Datumaro subsets.
_SPLIT_TO_SUBSET: dict[str, Subset] = {
    "train": Subset.TRAINING,
    "validation": Subset.VALIDATION,
    "test": Subset.TESTING,
}

# Rows read from each parquet at a time. The embedded images make the row groups
# memory-heavy, so a modest batch keeps peak memory bounded.
_BATCH_SIZE = 128

_HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{revision}/data/{filename}"


def _row_annotations(
    objects: dict[str, list],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Convert a single row's ``objects`` struct into Coco annotation arrays.

    Args:
        objects: The per-row ``objects`` value as produced by
            ``pyarrow``'s ``to_pylist`` — a dict of parallel lists with keys
            ``bbox`` (COCO ``[x, y, w, h]``), ``category`` and ``area``.

    Returns:
        A ``(bboxes, labels, areas, iscrowd)`` tuple. Every element is ``None``
        when the image has no annotations.
    """
    bbox = objects["bbox"]
    if not bbox:
        return None, None, None, None

    bboxes = np.asarray(bbox, dtype=np.float32)
    labels = np.asarray(objects["category"], dtype=np.int64)
    areas = np.asarray(objects["area"], dtype=np.float32)
    iscrowd = np.zeros(len(bboxes), dtype=np.int32)
    return bboxes, labels, areas, iscrowd


def _build_dataset(
    parquet_paths: Mapping[Subset, Path],
    label_names: Sequence[str],
    images_dir: Path,
) -> Dataset:
    """Materialise images from the parquet splits and build a ``CocoSample`` dataset."""
    images_dir.mkdir(parents=True, exist_ok=True)

    dataset: Dataset = Dataset(
        CocoSample,
        categories={"labels": CocoCategories(labels=tuple(label_names))},
    )

    image_id = 0
    # Iterate subsets in a fixed order so the resulting dataset is deterministic.
    for subset in (Subset.TRAINING, Subset.VALIDATION, Subset.TESTING):
        parquet_path = parquet_paths.get(subset)
        if parquet_path is None:
            continue

        parquet_file = pq.ParquetFile(str(parquet_path))
        for batch in parquet_file.iter_batches(batch_size=_BATCH_SIZE):
            images = batch.column("image").to_pylist()
            widths = batch.column("width").to_pylist()
            heights = batch.column("height").to_pylist()
            objects_col = batch.column("objects").to_pylist()

            for image_struct, width, height, objects in zip(images, widths, heights, objects_col, strict=True):
                img_bytes = image_struct["bytes"]
                suffix = (image_struct.get("path") or "image.jpg").rsplit(".", 1)[-1].lower()
                img_path = images_dir / f"img_{image_id:06d}.{suffix}"
                img_path.write_bytes(img_bytes)

                bboxes, labels, areas, iscrowd = _row_annotations(objects)

                dataset.append(
                    CocoSample(
                        image=LazyImage(img_path),
                        image_info=ImageInfo(width=int(width), height=int(height)),
                        image_id=image_id,
                        subset=subset,
                        bboxes=bboxes,
                        labels=labels,
                        polygons=None,
                        areas=areas,
                        iscrowd=iscrowd,
                        caption_group_ids=None,
                        captions=None,
                        keypoints=None,
                    ),
                )
                image_id += 1

    return dataset


def prepare_roboflow_hf_dataset(
    args: DatasetArgs,
    *,
    repo: str,
    revision: str,
    label_names: Sequence[str],
    split_files: Mapping[str, str],
) -> None:
    """Download an RF100 HuggingFace mirror and export it as a Datumaro dataset.

    Args:
        args: Parsed CLI arguments (``--output-dir`` / ``--name``).
        repo: HuggingFace dataset id, e.g. ``"Francesco/vehicles-q0x2v"``.
        revision: Pinned commit SHA of the dataset for reproducibility.
        label_names: Class names in ``ClassLabel`` index order. Index 0 is the
            Roboflow super-category placeholder that never appears in
            annotations; it is kept so the remaining indices line up with the
            parquet ``category`` ids.
        split_files: Mapping of upstream split name (``"train"`` /
            ``"validation"`` / ``"test"``) to its parquet file name.
    """
    # Download each split parquet into the shared archive directory.
    parquet_paths: dict[Subset, Path] = {}
    for split_name, filename in split_files.items():
        subset = _SPLIT_TO_SUBSET[split_name]
        url = _HF_RESOLVE.format(repo=repo, revision=revision, filename=filename)
        parquet_paths[subset] = download(
            url,
            dest_dir=args.archive_dir,
            filename=f"{args.name}_{split_name}.parquet",
        )

    # Staging directory for the JPEGs materialised from the parquet; ``args.dest``
    # will hold the final Datumaro dataset.
    staging = args.archive_dir / f"{args.name}_raw"
    images_dir = staging / "images"

    logger.info("Building Datumaro dataset from %s ...", repo)
    print(f"Building Datumaro dataset from {repo} ...")
    dataset = _build_dataset(parquet_paths, label_names, images_dir)
    print(f"  Dataset length: {len(dataset)}")

    # ``export_dataset`` requires that the output path does NOT exist yet,
    # so remove any leftover from a previous run and let it create the dir.
    if args.dest.exists():
        shutil.rmtree(args.dest)
    args.dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting dataset to {args.dest} ...")
    export_dataset(dataset, args.dest)

    # Clean up the downloaded parquet splits and staged images.
    for parquet_path in parquet_paths.values():
        parquet_path.unlink(missing_ok=True)
    shutil.rmtree(staging, ignore_errors=True)

    print(f"Dataset '{args.name}' ready at {args.dest}")
