# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXDetectionDataset."""

from __future__ import annotations

from otx.data.entity.sample import DetectionSample

from .base_new import OTXDataset


class OTXDetectionDataset(OTXDataset):
    """OTXDataset class for detection task using new Datumaro experimental Dataset."""

    def __init__(self, **kwargs) -> None:
        """Initialize _OTXDetectionDataset.

        Args:
            **kwargs: Keyword arguments to pass to OTXDataset
        """
        kwargs["sample_type"] = DetectionSample
        super().__init__(**kwargs)

    def get_idx_list_per_classes(self, use_string_label: bool = False) -> dict[int, list[int]]:
        """Get a dictionary mapping class labels (string or int) to lists of samples.

        Args:
            use_string_label (bool): If True, use string class labels as keys.
                If False, use integer indices as keys.
        """
        idx_list_per_classes: dict[int, list[int]] = {}
        for idx in range(len(self)):
            item = self.dm_subset[idx]
            labels = item.label.tolist()
            if use_string_label:
                labels = [self.label_info.label_names[label] for label in labels]
            for label in labels:
                if label not in idx_list_per_classes:
                    idx_list_per_classes[label] = []
                idx_list_per_classes[label].append(idx)
        return idx_list_per_classes


# class _OTXDetectionDataset(OTXDataset, DataAugSwitchMixin):  # type: ignore[misc]
#     """OTXDataset class for detection task."""
#
#     def _get_item_impl(self, index: int) -> OTXDataItem | None:
#         item = self.dm_subset[index]
#         img = item.media_as(Image)
#         ignored_labels: list[int] = []  # This should be assigned form item
#         img_data, img_shape, _ = self._get_img_data_and_shape(img)
#
#         bbox_anns = [ann for ann in item.annotations if isinstance(ann, Bbox)]
#
#         bboxes = (
#             np.stack([ann.points for ann in bbox_anns], axis=0).astype(np.float32)
#             if len(bbox_anns) > 0
#             else np.zeros((0, 4), dtype=np.float32)
#         )
#
#         entity = OTXDataItem(
#             image=img_data,
#             img_info=ImageInfo(
#                 img_idx=index,
#                 img_shape=img_shape,
#                 ori_shape=img_shape,
#                 image_color_channel=self.image_color_channel,
#                 ignored_labels=ignored_labels,
#             ),
#             bboxes=tv_tensors.BoundingBoxes(
#                 bboxes,
#                 format=tv_tensors.BoundingBoxFormat.XYXY,
#                 canvas_size=img_shape,
#                 dtype=torch.float32,
#             ),
#             label=torch.as_tensor([ann.label for ann in bbox_anns], dtype=torch.long),
#         )
#         # Apply augmentation switch if available
#         if self.has_dynamic_augmentation:
#             self._apply_augmentation_switch()
#
#         return self._apply_transforms(entity)
