# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from app.models.model_manifest import BenchmarkMetrics, MemoryFootprint, ModelStats
from app.models.system import DeviceInfo, DeviceType
from app.services.memory_estimation_service import (
    FRAMEWORK_OVERHEAD_MB,
    MEMORY_SAFETY_FRACTION,
    check_training_memory,
    compute_memory_footprint,
    estimate_training_memory_mb,
    recommend_lighter_models,
)


def _stats(trainable_parameters: float, gigaflops: float) -> ModelStats:
    return ModelStats(
        gigaflops=gigaflops,
        trainable_parameters=trainable_parameters,
        benchmark_metrics=BenchmarkMetrics(coco_map_50_95=40.0),
    )


class _Manifest:
    """Minimal stand-in for a ModelManifest used by the estimation helpers."""

    def __init__(
        self,
        manifest_id: str,
        name: str,
        footprint: MemoryFootprint,
        batch_size: int,
    ) -> None:
        self.id = manifest_id
        self.name = name
        self.memory_footprint = footprint

        class _Training:
            def __init__(self, bs: int) -> None:
                self.batch_size = bs
                self.input_size_width = 640
                self.input_size_height = 640

        class _Hyperparameters:
            def __init__(self, bs: int) -> None:
                self.training = _Training(bs)

        self.hyperparameters = _Hyperparameters(batch_size)


class TestComputeMemoryFootprint:
    def test_components_scale_as_expected(self):
        footprint = compute_memory_footprint(
            stats=_stats(trainable_parameters=9.0, gigaflops=26.8),
            input_size_width=640,
            input_size_height=640,
            reference_batch_size=8,
        )

        # Base = framework overhead + 9M params * 16 bytes.
        expected_base = FRAMEWORK_OVERHEAD_MB + (9.0e6 * 16) / (1024 * 1024)
        assert footprint.base_memory_mb == round(expected_base, 1)
        assert footprint.reference_batch_size == 8
        assert footprint.per_sample_memory_mb > 0
        # Total must equal base + per_sample * reference_batch_size.
        assert footprint.estimated_training_memory_mb == round(
            footprint.base_memory_mb + footprint.per_sample_memory_mb * 8, 1
        )

    def test_larger_input_increases_per_sample_memory(self):
        small = compute_memory_footprint(_stats(9.0, 26.8), 320, 320, 8)
        large = compute_memory_footprint(_stats(9.0, 26.8), 640, 640, 8)
        assert large.per_sample_memory_mb > small.per_sample_memory_mb


class TestEstimateTrainingMemory:
    def test_scales_linearly_with_batch_size(self):
        footprint = MemoryFootprint(
            reference_batch_size=8,
            estimated_training_memory_mb=2000.0,
            base_memory_mb=1000.0,
            per_sample_memory_mb=125.0,
        )
        assert estimate_training_memory_mb(footprint, 1) == 1125.0
        assert estimate_training_memory_mb(footprint, 8) == 2000.0
        assert estimate_training_memory_mb(footprint, 16) == 3000.0


class TestCheckTrainingMemory:
    def test_gpu_uses_device_memory(self):
        manifest = _Manifest(
            "m",
            "Model",
            MemoryFootprint(
                reference_batch_size=8,
                estimated_training_memory_mb=5000.0,
                base_memory_mb=4000.0,
                per_sample_memory_mb=125.0,
            ),
            batch_size=8,
        )
        device = DeviceInfo(type=DeviceType.XPU, name="GPU", memory=4 * 1024 * 1024 * 1024, index=0)

        result = check_training_memory(manifest, device, available_system_memory_mb=64000.0)

        assert result.estimated_memory_mb == 5000.0
        assert result.available_memory_mb == 4096.0
        assert result.usable_memory_mb == round(4096.0 * MEMORY_SAFETY_FRACTION, 1)
        assert result.fits is False

    def test_cpu_uses_system_memory(self):
        manifest = _Manifest(
            "m",
            "Model",
            MemoryFootprint(
                reference_batch_size=8,
                estimated_training_memory_mb=2000.0,
                base_memory_mb=1000.0,
                per_sample_memory_mb=125.0,
            ),
            batch_size=8,
        )
        device = DeviceInfo.cpu()

        result = check_training_memory(manifest, device, available_system_memory_mb=32000.0)

        assert result.available_memory_mb == 32000.0
        assert result.fits is True

    def test_cpu_does_not_fit_when_free_memory_is_low(self):
        # A model that easily fits in total RAM but not in the small amount currently free.
        manifest = _Manifest(
            "m",
            "Model",
            MemoryFootprint(
                reference_batch_size=8,
                estimated_training_memory_mb=3000.0,
                base_memory_mb=2000.0,
                per_sample_memory_mb=125.0,
            ),
            batch_size=8,
        )
        device = DeviceInfo.cpu()

        # Only ~1.5 GB free even though the machine may have 64 GB total.
        result = check_training_memory(manifest, device, available_system_memory_mb=1536.0)

        assert result.available_memory_mb == 1536.0
        assert result.fits is False


class TestRecommendLighterModels:
    def test_returns_only_fitting_models_sorted_by_descending_memory(self):
        device = DeviceInfo(type=DeviceType.XPU, name="GPU", memory=4 * 1024 * 1024 * 1024, index=0)
        heavy = _Manifest(
            "heavy",
            "Heavy",
            MemoryFootprint(
                reference_batch_size=8,
                estimated_training_memory_mb=9000.0,
                base_memory_mb=8000.0,
                per_sample_memory_mb=125.0,
            ),
            batch_size=8,
        )
        medium = _Manifest(
            "medium",
            "Medium",
            MemoryFootprint(
                reference_batch_size=8,
                estimated_training_memory_mb=3000.0,
                base_memory_mb=2000.0,
                per_sample_memory_mb=125.0,
            ),
            batch_size=8,
        )
        light = _Manifest(
            "light",
            "Light",
            MemoryFootprint(
                reference_batch_size=8,
                estimated_training_memory_mb=1200.0,
                base_memory_mb=1000.0,
                per_sample_memory_mb=25.0,
            ),
            batch_size=8,
        )

        recommendations = recommend_lighter_models(
            candidate_manifests=[heavy, medium, light],
            device=device,
            available_system_memory_mb=64000.0,
            exclude_id="heavy",
        )

        # heavy is excluded and does not fit; medium and light fit; most capable (largest) first.
        assert [r.id for r in recommendations] == ["medium", "light"]
