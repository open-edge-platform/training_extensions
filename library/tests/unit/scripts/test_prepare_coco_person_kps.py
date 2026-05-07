# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for ``scripts/benchmark_datasets/prepare_coco_person_kps.py``.

This test performs a real network download and is therefore marked with
``@pytest.mark.network``. It is skipped by default and only runs when pytest
is invoked with ``--run-network``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from datumaro.experimental.export_import import import_dataset
from datumaro.experimental.fields import Subset

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "benchmark_datasets" / "prepare_coco_person_kps.py"


@pytest.mark.network
def test_prepare_coco_person_kps_end_to_end(tmp_path: Path) -> None:
    """Run the prepare script end-to-end and validate the exported Datumaro dataset."""
    assert _SCRIPT.is_file(), f"Script not found: {_SCRIPT}"

    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--output-dir", str(tmp_path), "--name", "coco_person_kps"],
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 min: COCO val images are ~1GB
    )
    assert result.returncode == 0, (
        f"Script failed (exit {result.returncode})\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )

    dataset_dir = tmp_path / "coco_person_kps"
    assert dataset_dir.is_dir(), f"Expected output directory not found: {dataset_dir}"

    # Validate the exported dataset
    dataset = import_dataset(dataset_dir)
    # Should have ~2500-2800 images with person keypoints
    assert len(dataset) > 2000, f"Expected >2000 images, got {len(dataset)}"

    counts = {Subset.VALIDATION: 0, Subset.TESTING: 0}
    for sample in dataset:
        counts[sample.subset] = counts.get(sample.subset, 0) + 1

    assert counts[Subset.VALIDATION] > 0
    assert counts[Subset.TESTING] > 0
