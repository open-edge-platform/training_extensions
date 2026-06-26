# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``getitune.benchmark.roboflow_hf``.

These tests build a small synthetic parquet that mimics the Roboflow-100
HuggingFace mirror schema and run the conversion offline (no network), so they
execute as part of the default test suite.
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datumaro.experimental.categories import LabelCategories
from datumaro.experimental.export_import import import_dataset
from datumaro.experimental.fields import Subset
from PIL import Image

from getitune.benchmark import roboflow_hf
from getitune.benchmark.dataset_helpers import DatasetArgs
from getitune.data.entity.sample import DetectionSample

# Arrow type mirroring the RF100 ``objects`` struct (struct-of-arrays).
_OBJECTS_TYPE = pa.struct(
    [
        ("id", pa.list_(pa.int64())),
        ("area", pa.list_(pa.int64())),
        ("bbox", pa.list_(pa.list_(pa.float32(), 4))),
        ("category", pa.list_(pa.int64())),
    ],
)
_IMAGE_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])

_LABEL_NAMES = ("placeholder", "cat_a", "cat_b")


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    """Return the bytes of a tiny solid-colour JPEG."""
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="JPEG")
    return buf.getvalue()


def _write_parquet(path: Path, rows: list[dict]) -> None:
    """Write *rows* to a parquet file using the RF100 mirror schema."""
    table = pa.table(
        {
            "image_id": pa.array([r["image_id"] for r in rows], type=pa.int64()),
            "image": pa.array([r["image"] for r in rows], type=_IMAGE_TYPE),
            "width": pa.array([r["width"] for r in rows], type=pa.int32()),
            "height": pa.array([r["height"] for r in rows], type=pa.int32()),
            "objects": pa.array([r["objects"] for r in rows], type=_OBJECTS_TYPE),
        },
    )
    pq.write_table(table, path)


def _empty_objects() -> dict[str, list]:
    return {"id": [], "area": [], "bbox": [], "category": []}


@pytest.fixture
def synthetic_splits(tmp_path: Path) -> dict[str, Path]:
    """Create train/validation/test parquet files and return their paths."""
    src = tmp_path / "src"
    src.mkdir()

    img = _jpeg_bytes((120, 10, 10))

    train = [
        {
            "image_id": 0,
            "image": {"bytes": img, "path": "a.jpg"},
            "width": 8,
            "height": 8,
            "objects": {"id": [1], "area": [12], "bbox": [[1.0, 2.0, 3.0, 4.0]], "category": [1]},
        },
        {
            "image_id": 1,
            "image": {"bytes": img, "path": "b.jpg"},
            "width": 8,
            "height": 8,
            "objects": {
                "id": [2, 3],
                "area": [6, 9],
                "bbox": [[0.0, 0.0, 2.0, 3.0], [1.0, 1.0, 3.0, 3.0]],
                "category": [1, 2],
            },
        },
    ]
    validation = [
        {
            "image_id": 0,
            "image": {"bytes": img, "path": "c.png"},
            "width": 8,
            "height": 8,
            "objects": {"id": [4], "area": [4], "bbox": [[2.0, 2.0, 2.0, 2.0]], "category": [2]},
        },
    ]
    # An image with no annotations must be handled gracefully.
    test = [
        {
            "image_id": 0,
            "image": {"bytes": img, "path": "d.jpg"},
            "width": 8,
            "height": 8,
            "objects": _empty_objects(),
        },
    ]

    paths = {
        "train": src / "train.parquet",
        "validation": src / "validation.parquet",
        "test": src / "test.parquet",
    }
    _write_parquet(paths["train"], train)
    _write_parquet(paths["validation"], validation)
    _write_parquet(paths["test"], test)
    return paths


def test_prepare_roboflow_hf_dataset(
    tmp_path: Path,
    synthetic_splits: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The helper converts the synthetic parquet splits into a Datumaro dataset."""

    def fake_download(url: str, dest_dir: Path, filename: str | None = None) -> Path:
        assert filename is not None
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename
        for split, src in synthetic_splits.items():
            if filename.endswith(f"_{split}.parquet"):
                shutil.copy(src, dest)
                return dest
        msg = f"unexpected download filename: {filename}"
        raise AssertionError(msg)

    monkeypatch.setattr(roboflow_hf, "download", fake_download)

    output_dir = tmp_path / "out"
    args = DatasetArgs(output_dir=output_dir, name="synthetic")

    roboflow_hf.prepare_roboflow_hf_dataset(
        args,
        repo="Francesco/example",
        revision="deadbeef",
        label_names=_LABEL_NAMES,
        split_files={
            "train": "train-x.parquet",
            "validation": "validation-x.parquet",
            "test": "test-x.parquet",
        },
    )

    dataset_dir = output_dir / "synthetic"
    assert dataset_dir.is_dir()

    # Downloaded parquet splits and the staging dir must be cleaned up.
    archives = output_dir / ".archives"
    assert not (archives / "synthetic_train.parquet").exists()
    assert not (archives / "synthetic_raw").exists()

    dataset = import_dataset(dataset_dir)
    assert len(dataset) == 4  # 2 train + 1 val + 1 test

    label_categories = dataset.label_categories
    assert isinstance(label_categories, LabelCategories)
    assert tuple(label_categories.labels) == _LABEL_NAMES

    counts = {Subset.TRAINING: 0, Subset.VALIDATION: 0, Subset.TESTING: 0}
    for sample in dataset:
        counts[sample.subset] += 1
    assert counts == {Subset.TRAINING: 2, Subset.VALIDATION: 1, Subset.TESTING: 1}

    # The dataset must convert to the getitune detection schema (COCO xywh boxes).
    detection_dataset = dataset.convert_to_schema(DetectionSample)
    total_boxes = sum(0 if s.bboxes is None else len(s.bboxes) for s in detection_dataset)
    assert total_boxes == 4  # 1 + 2 + 1 + 0 across the four images
