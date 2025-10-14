# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXClassificationDatasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from datumaro import Image, Label
from datumaro.components.annotation import AnnotationType
from torch.nn import functional
from torchvision.transforms.v2.functional import to_dtype, to_image

from otx.data.dataset.base import OTXDataset, Transforms
from otx.data.entity.base import ImageInfo
from otx.data.entity.torch import OTXDataItem
from otx.types import OTXTaskType
from otx.types.image import ImageColorChannel
from otx.types.label import HLabelInfo

if TYPE_CHECKING:
    from datumaro import Dataset as DmDataset


class OTXMulticlassClsDataset(OTXDataset):
    """OTX Dataset for multi-class classification tasks.

    This dataset handles single-label classification where each image belongs to exactly one class.
    It processes Datumaro dataset items and converts them into OTXDataItem format suitable for
    multi-class classification training and inference.

    Args:
        dm_subset (DmDataset): Datumaro dataset subset containing the data items.
        transforms (Transforms, optional): Transformations to apply to the data.
        max_refetch (int): Maximum number of retries when fetching a data item fails.
        image_color_channel (ImageColorChannel): Color channel format for images (RGB, BGR, etc.).
        stack_images (bool): Whether to stack images in batch processing.
        to_tv_image (bool): Whether to convert images to torchvision format.
        data_format (str): Format of the source data (e.g., "arrow", "coco").

    Raises:
        ValueError: If an image has multiple labels (multi-label case).

    Example:
        >>> from otx.data.dataset.classification import OTXMulticlassClsDataset
        >>> dataset = OTXMulticlassClsDataset(
        ...     dm_subset=my_dm_subset,
        ...     transforms=my_transforms,
        ...     image_color_channel=ImageColorChannel.RGB
        ... )
        >>> item = dataset[0]  # Get first item
    """

    def __init__(
        self,
        dm_subset: DmDataset,
        transforms: Transforms | None = None,
        max_refetch: int = 1000,
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        stack_images: bool = True,
        to_tv_image: bool = True,
        data_format: str = "",
    ) -> None:
        super().__init__(
            dm_subset=dm_subset,
            transforms=transforms,
            max_refetch=max_refetch,
            image_color_channel=image_color_channel,
            stack_images=stack_images,
            to_tv_image=to_tv_image,
            data_format=data_format,
        )

    def _get_item_impl(self, index: int) -> OTXDataItem | None:
        """Get a single data item from the dataset.

        Args:
            index: Index of the item to retrieve.

        Returns:
            OTXDataItem or None: The processed data item with image and label tensors,
                or None if the item could not be processed.

        Raises:
            ValueError: If the image has multiple labels (not allowed for multi-class).
        """
        item = self.dm_subset[index]
        img = item.media_as(Image)
        roi = item.attributes.get("roi", None)
        img_data, img_shape, _ = self._get_img_data_and_shape(img, roi)
        image = to_dtype(to_image(img_data), dtype=torch.float32)
        if roi:
            # extract labels from ROI
            labels_ids = [
                label["label"]["_id"] for label in roi["labels"] if label["label"]["domain"] == "CLASSIFICATION"
            ]
            if self.data_format == "arrow":
                label_anns = [self.label_info.label_ids.index(label_id) for label_id in labels_ids]
            else:
                label_anns = [self.label_info.label_names.index(label_id) for label_id in labels_ids]
        else:
            # extract labels from annotations
            label_anns = [ann.label for ann in item.annotations if isinstance(ann, Label)]

        if len(label_anns) > 1:
            msg = f"Multi-class Classification can't use the multi-label, currently len(labels) = {len(label_anns)}"
            raise ValueError(msg)

        entity = OTXDataItem(
            image=image,
            label=torch.as_tensor(label_anns, dtype=torch.long),
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
                image_color_channel=self.image_color_channel,
            ),
        )
        return self._apply_transforms(entity)

    @property
    def task_type(self) -> OTXTaskType:
        """OTX Task Type for the dataset.

        Returns:
            OTXTaskType: The multi-class classification task type.
        """
        return OTXTaskType.MULTI_CLASS_CLS


class OTXMultilabelClsDataset(OTXDataset):
    """OTX Dataset for multi-label classification tasks.

    This dataset handles multi-label classification where each image can belong to multiple classes
    simultaneously. It processes Datumaro dataset items and converts them into OTXDataItem format
    with one-hot encoded labels suitable for multi-label classification training and inference.

    Args:
        dm_subset (DmDataset): Datumaro dataset subset containing the data items.
        transforms (Transforms, optional): Transform operations to apply to the data items.
        max_refetch (int): Maximum number of retries when fetching a data item fails.
        image_color_channel (ImageColorChannel): Color channel format for images (RGB, BGR, etc.).
        stack_images (bool): Whether to stack images in batch processing.
        to_tv_image (bool): Whether to convert images to torchvision format.
        data_format (str): Format of the source data (e.g., "arrow", "coco").

    Attributes:
        num_classes (int): Number of classes in the dataset.

    Example:
        >>> from otx.data.dataset.classification import OTXMultilabelClsDataset
        >>> dataset = OTXMultilabelClsDataset(
        ...     dm_subset=my_dm_subset,
        ...     transforms=my_transforms,
        ...     image_color_channel=ImageColorChannel.RGB
        ... )
        >>> item = dataset[0]  # Get first item with one-hot encoded labels
    """

    def __init__(
        self,
        dm_subset: DmDataset,
        transforms: Transforms | None = None,
        max_refetch: int = 1000,
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        stack_images: bool = True,
        to_tv_image: bool = True,
        data_format: str = "",
    ) -> None:
        super().__init__(
            dm_subset=dm_subset,
            transforms=transforms,
            max_refetch=max_refetch,
            image_color_channel=image_color_channel,
            stack_images=stack_images,
            to_tv_image=to_tv_image,
            data_format=data_format,
        )
        self.num_classes = len(self.dm_subset.categories()[AnnotationType.label])

    def _get_item_impl(self, index: int) -> OTXDataItem | None:
        """Get a single data item from the dataset.

        Args:
            index: Index of the item to retrieve.

        Returns:
            OTXDataItem or None: The processed data item with image and one-hot encoded label tensors,
                or None if the item could not be processed.
        """
        item = self.dm_subset[index]
        img = item.media_as(Image)
        ignored_labels: list[int] = []  # This should be assigned form item
        img_data, img_shape, _ = self._get_img_data_and_shape(img)
        img_data = to_dtype(to_image(img_data), dtype=torch.float32)

        label_ids = set()
        for ann in item.annotations:
            # multilabel information stored in 'multi_label_ids' attribute when the source format is arrow
            if "multi_label_ids" in ann.attributes:
                for lbl_idx in ann.attributes["multi_label_ids"]:
                    label_ids.add(lbl_idx)

            if isinstance(ann, Label):
                label_ids.add(ann.label)
            else:
                # If the annotation is not Label, it should be converted to Label.
                # For Chained Task: Detection (Bbox) -> Classification (Label)
                label = Label(label=ann.label)
                label_ids.add(label.label)
        labels = torch.as_tensor(list(label_ids))

        entity = OTXDataItem(
            image=img_data,
            label=self._convert_to_onehot(labels, ignored_labels),
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
                image_color_channel=self.image_color_channel,
                ignored_labels=ignored_labels,
            ),
        )
        return self._apply_transforms(entity)

    def _convert_to_onehot(self, labels: torch.tensor, ignored_labels: list[int]) -> torch.tensor:
        """Convert label to one-hot vector format.

        Args:
            labels: Input label tensor to convert.
            ignored_labels: List of label indices to ignore.

        Returns:
            torch.tensor: One-hot encoded label tensor where ignored labels are set to -1.
        """
        # Torch's one_hot() expects the input to be of type long
        # However, when labels are empty, they are of type float32
        onehot = functional.one_hot(labels.long(), self.num_classes).sum(0).clamp_max_(1)
        if ignored_labels:
            for ignore_label in ignored_labels:
                onehot[ignore_label] = -1
        return onehot

    @property
    def task_type(self) -> OTXTaskType:
        """OTX Task Type for the dataset.

        Returns:
            OTXTaskType: The multi-label classification task type.
        """
        return OTXTaskType.MULTI_LABEL_CLS


class OTXHlabelClsDataset(OTXDataset):
    """OTX Dataset for hierarchical label classification tasks.

    This dataset handles hierarchical classification where labels are organized in a tree structure
    with multiple classification heads. It supports both multiclass heads (where one class per head
    is selected) and multilabel heads (where multiple classes can be selected simultaneously).

    The dataset processes Datumaro dataset items and converts them into OTXDataItem format with
    hierarchical label encoding suitable for H-label classification training and inference.

    Args:
        dm_subset (DmDataset): Datumaro dataset subset containing the data items.
        transforms (Transforms, optional): Transform operations to apply to the data items.
        max_refetch (int): Maximum number of retries when fetching a data item fails.
        image_color_channel (ImageColorChannel): Color channel format for images (RGB, BGR, etc.).
        stack_images (bool): Whether to stack images in batch processing.
        to_tv_image (bool): Whether to convert images to torchvision format.
        data_format (str): Format of the source data (e.g., "arrow", "coco").

    Attributes:
        dm_categories (datumaro.components.annotation.LabelCategories): Datumaro label categories for the dataset.
        label_info (HLabelInfo): HLabelInfo containing hierarchical label structure information.
        id_to_name_mapping (dict[str, str]): Mapping from label IDs to label names.

    Raises:
        ValueError: If the number of multiclass heads is 0.
        TypeError: If label_info is not of type HLabelInfo.

    Example:
        >>> from otx.data.dataset.classification import OTXHlabelClsDataset
        >>> dataset = OTXHlabelClsDataset(
        ...     dm_subset=my_dm_subset,
        ...     transforms=my_transforms,
        ...     image_color_channel=ImageColorChannel.RGB
        ... )
        >>> item = dataset[0]  # Get first item with hierarchical labels
    """

    def __init__(
        self,
        dm_subset: DmDataset,
        transforms: Transforms | None = None,
        max_refetch: int = 1000,
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        stack_images: bool = True,
        to_tv_image: bool = True,
        data_format: str = "",
    ) -> None:
        super().__init__(
            dm_subset=dm_subset,
            transforms=transforms,
            max_refetch=max_refetch,
            image_color_channel=image_color_channel,
            stack_images=stack_images,
            to_tv_image=to_tv_image,
            data_format=data_format,
        )
        self.dm_categories = self.dm_subset.categories()[AnnotationType.label]

        # Hlabel classification used HLabelInfo to insert the HLabelData.
        if self.data_format == "arrow":
            # arrow format stores label IDs as names, have to deal with that here
            self.label_info = HLabelInfo.from_dm_label_groups_arrow(self.dm_categories)
        else:
            self.label_info = HLabelInfo.from_dm_label_groups(self.dm_categories)

        self.id_to_name_mapping = dict(zip(self.label_info.label_ids, self.label_info.label_names))
        self.id_to_name_mapping[""] = ""

        if self.label_info.num_multiclass_heads == 0:
            msg = "The number of multiclass heads should be larger than 0."
            raise ValueError(msg)

        if self.data_format != "arrow":
            for dm_item in self.dm_subset:
                self._add_ancestors(dm_item.annotations)

    def _add_ancestors(self, label_anns: list[Label]) -> None:
        """Add ancestors recursively if some label miss the ancestor information.

        If the label tree likes below,
        object - vehicle -- car
                         |- bus
                         |- truck
        And annotation = ['car'], it should be ['car', 'vehicle', 'object'], to include the ancestor.

        This function add the ancestors to the annotation if missing.

        Args:
            label_anns: List of label annotations to which ancestors will be added.
        """

        def _label_idx_to_name(idx: int) -> str:
            return self.dm_categories[idx].name

        def _label_name_to_idx(name: str) -> int:
            indices = [idx for idx, val in enumerate(self.label_info.label_names) if val == name]
            return indices[0]

        def _get_label_group_idx(label_name: str) -> int:
            if isinstance(self.label_info, HLabelInfo):
                if self.data_format == "arrow":
                    return self.label_info.class_to_group_idx[self.id_to_name_mapping[label_name]][0]
                return self.label_info.class_to_group_idx[label_name][0]
            msg = f"self.label_info should have HLabelInfo type, got {type(self.label_info)}"
            raise ValueError(msg)

        def _find_ancestor_recursively(label_name: str, ancestors: list) -> list[str]:
            _, dm_label_category = self.dm_categories.find(label_name)
            parent_name = dm_label_category.parent if dm_label_category else ""

            if parent_name != "":
                ancestors.append(parent_name)
                _find_ancestor_recursively(parent_name, ancestors)
            return ancestors

        def _get_all_label_names_in_anns(anns: list[Label]) -> list[str]:
            return [_label_idx_to_name(ann.label) for ann in anns]

        all_label_names = _get_all_label_names_in_anns(label_anns)
        ancestor_dm_labels = []
        for ann in label_anns:
            label_idx = ann.label
            label_name = _label_idx_to_name(label_idx)
            ancestors = _find_ancestor_recursively(label_name, [])

            for i, ancestor in enumerate(ancestors):
                if ancestor not in all_label_names:
                    ancestor_dm_labels.append(
                        Label(
                            label=_label_name_to_idx(ancestor),
                            id=len(label_anns) + i,
                            group=_get_label_group_idx(ancestor),
                        ),
                    )
        label_anns.extend(ancestor_dm_labels)

    def _get_item_impl(self, index: int) -> OTXDataItem | None:
        """Get a single data item from the dataset with hierarchical labels.

        Args:
            index: Index of the item to retrieve.

        Returns:
            OTXDataItem or None: The processed data item with image and hierarchical label tensors,
                or None if the item could not be processed.
        """
        item = self.dm_subset[index]
        img = item.media_as(Image)
        ignored_labels: list[int] = []  # This should be assigned form item
        img_data, img_shape, _ = self._get_img_data_and_shape(img)
        img_data = to_dtype(to_image(img_data), dtype=torch.float32)

        label_ids = set()
        for ann in item.annotations:
            # in h-cls scenario multilabel information stored in 'multi_label_ids' attribute
            if "multi_label_ids" in ann.attributes:
                for lbl_idx in ann.attributes["multi_label_ids"]:
                    label_ids.add(lbl_idx)

            if isinstance(ann, Label):
                label_ids.add(ann.label)
            else:
                # If the annotation is not Label, it should be converted to Label.
                # For Chained Task: Detection (Bbox) -> Classification (Label)
                label = Label(label=ann.label)
                label_ids.add(label.label)

        hlabel_labels = self._convert_label_to_hlabel_format([Label(label=idx) for idx in label_ids], ignored_labels)

        entity = OTXDataItem(
            image=img_data,
            label=torch.as_tensor(hlabel_labels),
            img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
                image_color_channel=self.image_color_channel,
                ignored_labels=ignored_labels,
            ),
        )
        return self._apply_transforms(entity)

    def _convert_label_to_hlabel_format(self, label_anns: list[Label], ignored_labels: list[int]) -> list[int]:
        """Convert format of the label to the h-label.

        It converts the label format to h-label format.
        Total length of result is sum of number of hierarchy and number of multilabel classes.

        i.e.
        Let's assume that we used the same dataset with example of the definition of HLabelData
        and the original labels are ["Rigid", "Triangle", "Lion"].

        Then, h-label format will be [0, 1, 1, 0].
        The first N-th indices represent the label index of multiclass heads (N=num_multiclass_heads),
        others represent the multilabel labels.

        [Multiclass Heads]
        0-th index = 0 -> ["Rigid"(O), "Non-Rigid"(X)] <- First multiclass head
        1-st index = 1 -> ["Rectangle"(O), "Triangle"(X), "Circle"(X)] <- Second multiclass head

        [Multilabel Head]
        2, 3 indices = [1, 0] -> ["Lion"(O), "Panda"(X)]

        Args:
            label_anns: List of label annotations to convert.
            ignored_labels: List of label indices to ignore.

        Returns:
            list[int]: H-label formatted list where first N indices are multiclass heads
                and remaining indices are multilabel classes.

        Raises:
            TypeError: If label_info is not of type HLabelInfo.
        """
        if not isinstance(self.label_info, HLabelInfo):
            msg = f"The type of label_info should be HLabelInfo, got {type(self.label_info)}."
            raise TypeError(msg)

        num_multiclass_heads = self.label_info.num_multiclass_heads
        num_multilabel_classes = self.label_info.num_multilabel_classes

        class_indices = [0] * (num_multiclass_heads + num_multilabel_classes)
        for i in range(num_multiclass_heads):
            class_indices[i] = -1

        for ann in label_anns:
            if self.data_format == "arrow":
                # skips unknown labels for instance, the empty one
                if self.dm_categories.items[ann.label].name not in self.id_to_name_mapping:
                    continue
                ann_name = self.id_to_name_mapping[self.dm_categories.items[ann.label].name]
            else:
                ann_name = self.dm_categories.items[ann.label].name
            group_idx, in_group_idx = self.label_info.class_to_group_idx[ann_name]

            if group_idx < num_multiclass_heads:
                class_indices[group_idx] = in_group_idx
            elif ann.label not in ignored_labels:
                class_indices[num_multiclass_heads + in_group_idx] = 1
            else:
                class_indices[num_multiclass_heads + in_group_idx] = -1

        return class_indices

    @property
    def task_type(self) -> OTXTaskType:
        """OTX Task Type for the dataset.

        Returns:
            OTXTaskType: The hierarchical label classification task type.
        """
        return OTXTaskType.H_LABEL_CLS
