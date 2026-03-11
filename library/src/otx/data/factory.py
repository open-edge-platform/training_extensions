# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory classes for dataset and transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING

from otx.types.task import OTXTaskType
from otx.types.transformer_libs import TransformLibType

from .dataset.base import OTXDataset, Transforms

if TYPE_CHECKING:
    from datumaro.experimental import Dataset

    from otx.config.data import SubsetConfig


__all__ = ["OTXDatasetFactory", "TransformLibFactory"]


class TransformLibFactory:
    """Factory class for transform."""

    @classmethod
    def generate(cls: type[TransformLibFactory], config: SubsetConfig) -> Transforms:
        """Create transforms from factory."""
        if config.transform_lib_type == TransformLibType.TORCHVISION:
            from .transform_libs.torchvision import TorchVisionTransformLib

            return TorchVisionTransformLib.generate(config)

        raise NotImplementedError(config.transform_lib_type)


class OTXDatasetFactory:
    """Factory class for OTXDataset."""

    @classmethod
    def create(
        cls,
        task: OTXTaskType,
        dm_subset: Dataset,
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
