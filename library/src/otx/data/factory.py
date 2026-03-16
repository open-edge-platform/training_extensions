# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory classes for dataset and transforms."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from otx.types.task import OTXTaskType

from .augmentation.pipeline import CPUAugmentationPipeline
from .dataset.base import OTXDataset, Transforms

if TYPE_CHECKING:
    from datumaro.experimental import Dataset

    from otx.config.data import SubsetConfig

logger = logging.getLogger(__name__)


__all__ = ["OTXDatasetFactory", "TransformLibFactory"]


class TransformLibFactory:
    """Factory class for transform."""

    @classmethod
    def generate(cls: type[TransformLibFactory], config: SubsetConfig) -> Transforms | CPUAugmentationPipeline:
        """Create transforms from factory.

        Args:
            config: SubsetConfig with augmentations_cpu.

        Returns:
            CPUAugmentationPipeline built from config.
        """
        if config.augmentations_cpu:
            # Already a pipeline object (e.g., from from_file method)
            if isinstance(config.augmentations_cpu, CPUAugmentationPipeline):
                return config.augmentations_cpu
            return CPUAugmentationPipeline.from_config(config)

        # GPU-only configs may have an empty augmentations_cpu list;
        # return an identity pipeline so downstream code always gets a valid object.
        return CPUAugmentationPipeline(augmentations=[])


class OTXDatasetFactory:
    """Factory class for OTXDataset."""

    @classmethod
    def create(
        cls,
        task: OTXTaskType,
        dm_subset: Dataset,
        cfg_subset: SubsetConfig,
        # TODO(gdlg): Add support for ignore_index again
        ignore_index: int = 255,  # noqa: ARG003
    ) -> OTXDataset:
        """Create OTXDataset."""
        transforms = TransformLibFactory.generate(cfg_subset)

        # Auto-detect storage dtype from the first image's file header.
        # Reads only metadata (e.g. PNG IHDR), no pixel data is decoded.
        storage_dtype = cls._detect_storage_dtype(dm_subset)

        common_kwargs = {
            "dm_subset": dm_subset,
            "transforms": transforms,
            "storage_dtype": storage_dtype,
        }

        match task:
            case OTXTaskType.MULTI_CLASS_CLS:
                from .dataset.classification import OTXMulticlassClsDataset

                return OTXMulticlassClsDataset(**common_kwargs)

            case OTXTaskType.MULTI_LABEL_CLS:
                from .dataset.classification import OTXMultilabelClsDataset

                return OTXMultilabelClsDataset(**common_kwargs)

            case OTXTaskType.H_LABEL_CLS:
                from .dataset.classification import OTXHlabelClsDataset

                return OTXHlabelClsDataset(**common_kwargs)

            case OTXTaskType.DETECTION:
                from .dataset.detection import OTXDetectionDataset

                return OTXDetectionDataset(**common_kwargs)

            case OTXTaskType.ROTATED_DETECTION | OTXTaskType.INSTANCE_SEGMENTATION:
                from .dataset.instance_segmentation import OTXInstanceSegDataset

                return OTXInstanceSegDataset(task_type=task, **common_kwargs)

            case OTXTaskType.SEMANTIC_SEGMENTATION:
                from .dataset.segmentation import OTXSegmentationDataset

                return OTXSegmentationDataset(**common_kwargs)

            case OTXTaskType.KEYPOINT_DETECTION:
                from .dataset.keypoint_detection import OTXKeypointDetectionDataset

                return OTXKeypointDetectionDataset(**common_kwargs)

            case _:
                raise NotImplementedError(task)

    @staticmethod
    def _detect_storage_dtype(dm_subset: Dataset) -> str:
        """Probe the first image's file header to detect its bit depth.

        Uses ``PIL.Image.open`` which reads **only the file header** —
        no pixel data is decoded, so this is essentially free.

        Returns:
            ``"uint8"``, ``"uint16"``, or ``"float32"``.
        """
        from otx.data.entity.utils import detect_image_dtype

        try:
            first_item = next(iter(dm_subset))
            path = getattr(first_item.media, "path", None) if hasattr(first_item, "media") else None
            if path is not None:
                return detect_image_dtype(path)
        except (StopIteration, Exception):  # noqa: S110
            pass
        return "uint8"
