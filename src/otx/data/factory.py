# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory classes for dataset and transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING

from datumaro.experimental.categories import LabelCategories, LabelCategory, LabelGroup

from otx.types.image import ImageColorChannel
from otx.types.task import OTXTaskType
from otx.types.transformer_libs import TransformLibType

from .dataset.base import OTXDataset, Transforms
from .dataset.base_new import OTXDataset as OTXDatasetNew
from datumaro.experimental import Dataset as DatasetNew

from otx import LabelInfo, NullLabelInfo

from datumaro.components.dataset import Dataset as DmDataset
from datumaro.components.annotation import AnnotationType

if TYPE_CHECKING:
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
        ignore_index: int = 255,
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
            from .dataset.classification_new import OTXMulticlassClsDataset, ClassificationSample
            if isinstance(dm_subset, DmDataset):
                categories = cls._get_label_categories(dm_subset, data_format)
                dataset = DatasetNew(ClassificationSample, categories={"label": categories})
                for item in dm_subset:
                    dataset.append(ClassificationSample.from_dm_item(item))
                common_kwargs["dm_subset"] = dataset
            return OTXMulticlassClsDataset(**common_kwargs)

        if task == OTXTaskType.MULTI_LABEL_CLS:
            from otx.data.dataset.classification import OTXMultilabelClsDataset

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

    @staticmethod
    def _get_label_categories(dm_subset: DmDataset, data_format: str) -> LabelCategories:
        # TODO: Support hierarchical labels

        if dm_subset.categories() and data_format == "arrow":
            label_info = LabelInfo.from_dm_label_groups_arrow(dm_subset.categories()[AnnotationType.label])
        elif dm_subset.categories():
            label_info = LabelInfo.from_dm_label_groups(dm_subset.categories()[AnnotationType.label])
        else:
            label_info = NullLabelInfo()

        label_categories = [LabelCategory(name=label_name) for label_name in label_info.label_names]
        label_group = LabelGroup(name="default", labels=[name for name in label_info.label_names])
        return LabelCategories(items=label_categories, label_groups=[label_group])