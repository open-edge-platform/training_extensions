# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory classes for dataset and transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING

from datumaro.components.annotation import AnnotationType
from datumaro.experimental import Dataset as DatasetNew
from datumaro.experimental.categories import LabelCategories
from datumaro.experimental.legacy import convert_from_legacy

from otx import LabelInfo, NullLabelInfo
from otx.types.image import ImageColorChannel
from otx.types.task import OTXTaskType
from otx.types.transformer_libs import TransformLibType

from .dataset.base import OTXDataset, Transforms
from .dataset.base_new import OTXDataset as OTXDatasetNew

if TYPE_CHECKING:
    from datumaro.components.dataset import Dataset as DmDataset

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
        cls,
        task: OTXTaskType,
        dm_subset: DmDataset | DatasetNew,
        cfg_subset: SubsetConfig,
        data_format: str,
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        include_polygons: bool = False,
        # TODO(gdlg): Add support for ignore_index again
        ignore_index: int = 255,  # noqa: ARG003
    ) -> OTXDataset | OTXDatasetNew:
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
            from .dataset.classification_new import ClassificationSample, OTXMulticlassClsDataset

            categories = cls._get_label_categories(dm_subset, data_format)
            dataset = DatasetNew(ClassificationSample, categories={"label": categories})
            for item in dm_subset:
                if len(item.media.data.shape) == 3:  # TODO(albert): Account for grayscale images
                    dataset.append(ClassificationSample.from_dm_item(item))
            common_kwargs["dm_subset"] = dataset
            return OTXMulticlassClsDataset(**common_kwargs)

        if task == OTXTaskType.MULTI_LABEL_CLS:
            from .dataset.classification import OTXMultilabelClsDataset

            return OTXMultilabelClsDataset(**common_kwargs)

        if task == OTXTaskType.H_LABEL_CLS:
            from .dataset.classification import OTXHlabelClsDataset

            return OTXHlabelClsDataset(**common_kwargs)

        if task == OTXTaskType.DETECTION:
            from .dataset.detection_new import OTXDetectionDataset

            dataset = convert_from_legacy(dm_subset)
            common_kwargs["dm_subset"] = dataset

            return OTXDetectionDataset(**common_kwargs)

        if task in [OTXTaskType.ROTATED_DETECTION, OTXTaskType.INSTANCE_SEGMENTATION]:
            from .dataset.instance_segmentation_new import OTXInstanceSegDataset

            dataset = convert_from_legacy(dm_subset)
            common_kwargs["dm_subset"] = dataset

            return OTXInstanceSegDataset(include_polygons=include_polygons, **common_kwargs)

        if task == OTXTaskType.SEMANTIC_SEGMENTATION:
            from .dataset.segmentation_new import OTXSegmentationDataset

            dataset = convert_from_legacy(dm_subset)
            common_kwargs["dm_subset"] = dataset

            return OTXSegmentationDataset(**common_kwargs)

        if task == OTXTaskType.KEYPOINT_DETECTION:
            from .dataset.keypoint_detection_new import OTXKeypointDetectionDataset

            dataset = convert_from_legacy(dm_subset)
            common_kwargs["dm_subset"] = dataset
            return OTXKeypointDetectionDataset(**common_kwargs)

        raise NotImplementedError(task)

    @staticmethod
    def _get_label_categories(dm_subset: DmDataset, data_format: str) -> LabelCategories:
        if dm_subset.categories() and data_format == "arrow":
            label_info = LabelInfo.from_dm_label_groups_arrow(dm_subset.categories()[AnnotationType.label])
        elif dm_subset.categories():
            label_info = LabelInfo.from_dm_label_groups(dm_subset.categories()[AnnotationType.label])
        else:
            label_info = NullLabelInfo()
        return LabelCategories(labels=label_info.label_names)
