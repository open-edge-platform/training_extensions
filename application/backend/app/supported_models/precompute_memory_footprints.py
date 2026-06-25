# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Precompute per-model training-memory footprints and store them in the manifest YAML files.

For every model manifest this script computes a heuristic memory footprint (see
``app.services.memory_estimation_service``) at the model's configured batch size and writes a
``memory_footprint`` block into the model-specific YAML file. Re-running the script refreshes the
stored values in place (the block is replaced, not duplicated).

Run with::

    python -m app.supported_models.precompute_memory_footprints
"""

import re
from pathlib import Path

import yaml

from app.models.model_manifest import MemoryFootprint
from app.services.memory_estimation_service import compute_memory_footprint
from app.services.model_manifest_service import ModelManifestService

_MANIFESTS_DIR = Path(__file__).parent / "manifests"
_BLOCK_PATTERN = re.compile(r"\n*memory_footprint:\n(?:[ \t]+.*\n?)*\Z")


def _format_block(footprint: MemoryFootprint) -> str:
    """Render a ``memory_footprint`` YAML block for appending to a manifest file."""
    return (
        "\nmemory_footprint:\n"
        f"  reference_batch_size: {footprint.reference_batch_size}\n"
        f"  estimated_training_memory_mb: {footprint.estimated_training_memory_mb}\n"
        f"  base_memory_mb: {footprint.base_memory_mb}\n"
        f"  per_sample_memory_mb: {footprint.per_sample_memory_mb}\n"
    )


def precompute_memory_footprints() -> int:
    """Compute and persist memory footprints for all model manifests.

    Returns:
        int: The number of manifest files that were updated.
    """
    manifests_by_id = ModelManifestService.get_model_manifests()
    updated = 0

    for task_dir in sorted(d for d in _MANIFESTS_DIR.iterdir() if d.is_dir()):
        for model_file in sorted(task_dir.glob("*.yaml")):
            if model_file.name == "base.yaml":
                continue

            raw = yaml.safe_load(model_file.read_text(encoding="utf-8")) or {}
            model_id = raw.get("id")
            manifest = manifests_by_id.get(model_id) if model_id is not None else None
            if manifest is None:
                continue

            training = manifest.hyperparameters.training
            footprint = compute_memory_footprint(
                stats=manifest.stats,
                input_size_width=training.input_size_width,
                input_size_height=training.input_size_height,
                reference_batch_size=training.batch_size,
            )

            text = model_file.read_text(encoding="utf-8")
            text = _BLOCK_PATTERN.sub("", text).rstrip("\n")
            text = f"{text}\n{_format_block(footprint)}"
            model_file.write_text(text, encoding="utf-8")
            updated += 1

    return updated


if __name__ == "__main__":
    count = precompute_memory_footprints()
    print(f"Updated memory footprints in {count} manifest file(s).")
