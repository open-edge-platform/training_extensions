# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory classes for dataset and transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING

from datumaro.experimental.legacy import convert_from_legacy

from otx.types.task import OTXTaskType

from .augmentation.pipeline import CPUAugmentationPipeline
from .dataset.base import OTXDataset, Transforms

if TYPE_CHECKING:
    from datumaro.components.dataset import Dataset as DmDataset
    from datumaro.experimental import Dataset as DatasetNew

    from otx.config.data import SubsetConfig


__all__ = ["OTXDatasetFactory", "TransformLibFactory"]


class TransformLibFactory:
    """Factory class for transform.

    This factory supports both legacy transforms and new augmentations_cpu field.
    Priority: augmentations_cpu > transforms (legacy).
    """

    @classmethod
    def generate(cls: type[TransformLibFactory], config: SubsetConfig) -> Transforms | CPUAugmentationPipeline:
        """Create transforms from factory.

        If config.augmentations_cpu is set, uses the new CPUAugmentationPipeline.
        Otherwise falls back to legacy TorchVisionTransformLib for backward compatibility.

        Args:
            config: SubsetConfig with transforms or augmentations_cpu.

        Returns:
            Either CPUAugmentationPipeline (new) or Compose (legacy).
        """
        # New path: use augmentations_cpu if provided
        if config.augmentations_cpu:
            # Already a pipeline object (e.g., from from_file method)
            if isinstance(config.augmentations_cpu, CPUAugmentationPipeline):
                return config.augmentations_cpu
            return CPUAugmentationPipeline.from_config(config)

        raise NotImplementedError(config.transform_lib_type)


class OTXDatasetFactory:
    """Factory class for OTXDataset."""

    @classmethod
    def create(
        cls,
        task: OTXTaskType,
        dm_subset: DmDataset | DatasetNew,
        cfg_subset: SubsetConfig,
        data_format: str,
        # TODO(gdlg): Add support for ignore_index again
        ignore_index: int = 255,  # noqa: ARG003
    ) -> OTXDataset:
        """Create OTXDataset."""
        transforms = TransformLibFactory.generate(cfg_subset)
        common_kwargs = {
            "dm_subset": dm_subset,
            "transforms": transforms,
            "data_format": data_format,
            "to_tv_image": cfg_subset.to_tv_image,
        }

        if task == OTXTaskType.MULTI_CLASS_CLS:
            from .dataset.classification import OTXMulticlassClsDataset

            dataset = convert_from_legacy(dm_subset)
            common_kwargs["dm_subset"] = dataset
            return OTXMulticlassClsDataset(**common_kwargs)

        if task == OTXTaskType.MULTI_LABEL_CLS:
            from .dataset.classification import OTXMultilabelClsDataset

            dataset = convert_from_legacy(dm_subset, multi_label=True)
            common_kwargs["dm_subset"] = dataset
            return OTXMultilabelClsDataset(**common_kwargs)

        if task == OTXTaskType.H_LABEL_CLS:
            from .dataset.classification import OTXHlabelClsDataset

            dataset = convert_from_legacy(dm_subset, hierarchical=True)
            common_kwargs["dm_subset"] = dataset
            return OTXHlabelClsDataset(**common_kwargs)

        if task == OTXTaskType.DETECTION:
            from .dataset.detection import OTXDetectionDataset

            dataset = convert_from_legacy(dm_subset)
            common_kwargs["dm_subset"] = dataset
            return OTXDetectionDataset(**common_kwargs)

        if task in [OTXTaskType.ROTATED_DETECTION, OTXTaskType.INSTANCE_SEGMENTATION]:
            from .dataset.instance_segmentation import OTXInstanceSegDataset

            dataset = convert_from_legacy(dm_subset)
            common_kwargs["dm_subset"] = dataset
            return OTXInstanceSegDataset(task_type=task, **common_kwargs)

        if task == OTXTaskType.SEMANTIC_SEGMENTATION:
            from .dataset.segmentation import OTXSegmentationDataset

            dataset = convert_from_legacy(dm_subset)
            common_kwargs["dm_subset"] = dataset
            return OTXSegmentationDataset(**common_kwargs)

        if task == OTXTaskType.KEYPOINT_DETECTION:
            from .dataset.keypoint_detection import OTXKeypointDetectionDataset

            dataset = convert_from_legacy(dm_subset)
            common_kwargs["dm_subset"] = dataset
            return OTXKeypointDetectionDataset(**common_kwargs)

        raise NotImplementedError(task)
