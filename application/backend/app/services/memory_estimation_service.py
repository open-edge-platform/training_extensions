# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Heuristic estimation of the memory required to train a model.

The estimates produced here are deliberately conservative approximations derived from the model
statistics (parameter count, gigaflops and input resolution). They are meant to power a pre-flight
feasibility check that warns the user before a training job is started, not to predict the exact
peak memory usage of a particular run.
"""

from dataclasses import dataclass

from app.models.model_manifest import MemoryFootprint, ModelManifest, ModelStats
from app.models.system import DeviceInfo, DeviceType

# Bytes of state held per trainable parameter during training:
# weights (4) + gradients (4) + Adam optimizer first/second moments (4 + 4) = 16 bytes (FP32).
TRAINING_BYTES_PER_PARAM = 16

# Conservative fixed framework / device-context overhead (CUDA/XPU runtime, cuDNN workspaces, the
# Python process itself, dataloader workers, ...), in megabytes.
FRAMEWORK_OVERHEAD_MB = 1024.0

# Activation memory is approximated from the model's compute cost (gigaflops) scaled by the input
# resolution relative to a 640x640 reference. The coefficient converts this proxy into megabytes of
# activations per sample in a batch and was chosen to stay on the conservative (high) side.
ACTIVATION_FLOPS_COEFF = 6.0
REFERENCE_INPUT_PIXELS = 640 * 640

# Fraction of the available memory that a training run is allowed to occupy. Leaving headroom avoids
# OOM failures caused by fragmentation and transient peaks that the heuristic does not model.
MEMORY_SAFETY_FRACTION = 0.85

_BYTES_PER_MB = 1024 * 1024


def compute_memory_footprint(
    stats: ModelStats,
    input_size_width: int,
    input_size_height: int,
    reference_batch_size: int,
) -> MemoryFootprint:
    """Compute a heuristic training-memory footprint for a model.

    Args:
        stats: Model statistics (parameter count in millions and gigaflops).
        input_size_width: Training input width in pixels.
        input_size_height: Training input height in pixels.
        reference_batch_size: Batch size at which the total estimate is reported.

    Returns:
        MemoryFootprint: The batch-independent and per-sample memory components together with the
        total estimate at ``reference_batch_size``.
    """
    parameters = stats.trainable_parameters * 1e6
    base_memory_mb = FRAMEWORK_OVERHEAD_MB + (parameters * TRAINING_BYTES_PER_PARAM) / _BYTES_PER_MB

    input_pixels = input_size_width * input_size_height
    per_sample_memory_mb = ACTIVATION_FLOPS_COEFF * stats.gigaflops * (input_pixels / REFERENCE_INPUT_PIXELS)

    estimated_training_memory_mb = base_memory_mb + per_sample_memory_mb * reference_batch_size

    return MemoryFootprint(
        reference_batch_size=reference_batch_size,
        estimated_training_memory_mb=round(estimated_training_memory_mb, 1),
        base_memory_mb=round(base_memory_mb, 1),
        per_sample_memory_mb=round(per_sample_memory_mb, 3),
    )


def estimate_training_memory_mb(footprint: MemoryFootprint, batch_size: int) -> float:
    """Re-scale a precomputed footprint to a given batch size.

    Args:
        footprint: Precomputed memory footprint of the model.
        batch_size: Batch size that will actually be used for training.

    Returns:
        float: Estimated peak training memory in megabytes.
    """
    return footprint.base_memory_mb + footprint.per_sample_memory_mb * batch_size


def get_manifest_footprint(manifest: ModelManifest) -> MemoryFootprint:
    """Return the manifest's stored footprint, computing it on the fly when absent.

    Args:
        manifest: The model manifest.

    Returns:
        MemoryFootprint: The (possibly freshly computed) memory footprint.
    """
    if manifest.memory_footprint is not None:
        return manifest.memory_footprint
    training = manifest.hyperparameters.training
    return compute_memory_footprint(
        stats=manifest.stats,
        input_size_width=training.input_size_width,
        input_size_height=training.input_size_height,
        reference_batch_size=training.batch_size,
    )


@dataclass(frozen=True)
class MemoryCheckResult:
    """Outcome of a training-memory feasibility check."""

    fits: bool
    estimated_memory_mb: float
    available_memory_mb: float
    usable_memory_mb: float
    device_name: str


def check_training_memory(
    manifest: ModelManifest,
    device: DeviceInfo,
    total_system_memory_mb: float,
    batch_size: int | None = None,
) -> MemoryCheckResult:
    """Check whether a model can be trained on the given device within the available memory.

    Args:
        manifest: Manifest of the model to be trained.
        device: Device selected for training.
        total_system_memory_mb: Total host RAM in megabytes (used for CPU/AUTO training).
        batch_size: Batch size to estimate for; defaults to the manifest's configured batch size.

    Returns:
        MemoryCheckResult: The estimate, the available memory and whether training is expected to fit.
    """
    footprint = get_manifest_footprint(manifest)
    effective_batch_size = batch_size if batch_size is not None else manifest.hyperparameters.training.batch_size
    estimated_memory_mb = estimate_training_memory_mb(footprint, effective_batch_size)

    if device.type in (DeviceType.XPU, DeviceType.CUDA) and device.memory is not None:
        available_memory_mb = device.memory / _BYTES_PER_MB
    else:
        # CPU and AUTO training draws from host RAM.
        available_memory_mb = total_system_memory_mb

    usable_memory_mb = available_memory_mb * MEMORY_SAFETY_FRACTION

    return MemoryCheckResult(
        fits=estimated_memory_mb <= usable_memory_mb,
        estimated_memory_mb=round(estimated_memory_mb, 1),
        available_memory_mb=round(available_memory_mb, 1),
        usable_memory_mb=round(usable_memory_mb, 1),
        device_name=device.name,
    )


@dataclass(frozen=True)
class RecommendedModel:
    """A lighter model architecture that is expected to fit within the available memory."""

    id: str
    name: str
    estimated_memory_mb: float


def recommend_lighter_models(
    candidate_manifests: list[ModelManifest],
    device: DeviceInfo,
    total_system_memory_mb: float,
    exclude_id: str,
) -> list[RecommendedModel]:
    """Find model architectures of the same task that are expected to fit in memory.

    Args:
        candidate_manifests: Manifests to consider (typically all manifests for the project's task).
        device: Device selected for training.
        total_system_memory_mb: Total host RAM in megabytes.
        exclude_id: Architecture ID to exclude (the one that did not fit).

    Returns:
        list[RecommendedModel]: Fitting architectures sorted by descending estimated memory, so that
        the most capable model that still fits comes first.
    """
    recommendations: list[RecommendedModel] = []
    for manifest in candidate_manifests:
        if manifest.id == exclude_id:
            continue
        result = check_training_memory(
            manifest=manifest,
            device=device,
            total_system_memory_mb=total_system_memory_mb,
        )
        if result.fits:
            recommendations.append(
                RecommendedModel(
                    id=manifest.id,
                    name=manifest.name,
                    estimated_memory_mb=result.estimated_memory_mb,
                )
            )
    recommendations.sort(key=lambda model: model.estimated_memory_mb, reverse=True)
    return recommendations
