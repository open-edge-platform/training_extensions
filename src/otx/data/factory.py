# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory classes for dataset and transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING

from otx.types.image import ImageColorChannel
from otx.types.task import OTXTaskType
from otx.types.transformer_libs import TransformLibType

from .dataset.base import OTXDataset, Transforms

if TYPE_CHECKING:
    from datumaro import Dataset as DmDataset

    from otx.config.data import SubsetConfig


__all__ = ["TransformLibFactory", "OTXDatasetFactory"]


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
        cls: type[OTXDatasetFactory],
        task: OTXTaskType,
        dm_subset: DmDataset,
        cfg_subset: SubsetConfig,
        data_format: str,
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        include_polygons: bool = False,
        ignore_index: int = 255,
    ) -> OTXDataset:
        """Create OTXDataset."""
        transforms = TransformLibFactory.generate(cfg_subset)
        common_kwargs = {
            "dm_subset": dm_subset,
            "transforms": transforms,
            "data_format": data_format,
            "image_color_channel": image_color_channel,
            "to_tv_image": cfg_subset.to_tv_image,
        }

        if task in (
            OTXTaskType.ANOMALY,
            OTXTaskType.ANOMALY_CLASSIFICATION,
            OTXTaskType.ANOMALY_DETECTION,
            OTXTaskType.ANOMALY_SEGMENTATION,
        ):
            from .dataset.anomaly import OTXAnomalyDataset

            return OTXAnomalyDataset(task_type=task, **common_kwargs)

        if task == OTXTaskType.MULTI_CLASS_CLS:
            from .dataset.classification import OTXMulticlassClsDataset

            return OTXMulticlassClsDataset(**common_kwargs)

        if task == OTXTaskType.MULTI_LABEL_CLS:
            from .dataset.classification import OTXMultilabelClsDataset

            return OTXMultilabelClsDataset(**common_kwargs)

        if task == OTXTaskType.H_LABEL_CLS:
            from .dataset.classification import OTXHlabelClsDataset

            return OTXHlabelClsDataset(**common_kwargs)

        if task == OTXTaskType.DETECTION:
            from .dataset.detection import OTXDetectionDataset

            return OTXDetectionDataset(**common_kwargs)

        if task in [OTXTaskType.ROTATED_DETECTION, OTXTaskType.INSTANCE_SEGMENTATION]:
            from .dataset.instance_segmentation import OTXInstanceSegDataset

            return OTXInstanceSegDataset(include_polygons=include_polygons, **common_kwargs)

        if task == OTXTaskType.SEMANTIC_SEGMENTATION:
            from .dataset.segmentation import OTXSegmentationDataset

            return OTXSegmentationDataset(ignore_index=ignore_index, **common_kwargs)

        if task == OTXTaskType.KEYPOINT_DETECTION:
            from .dataset.keypoint_detection import OTXKeypointDetectionDataset

            return OTXKeypointDetectionDataset(**common_kwargs)

        raise NotImplementedError(task)
