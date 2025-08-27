# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXClassificationDatasets using new Datumaro experimental Dataset."""

from __future__ import annotations

from .base_new import OTXDataset
from ..entity.sample import ClassificationSample


class OTXMulticlassClsDataset(OTXDataset):
    """OTXDataset class for multi-class classification task using new Datumaro experimental Dataset."""

    def __init__(self, **kwargs) -> None:
        """Initialize OTXMulticlassClsDataset.

        Args:
            **kwargs: Keyword arguments to pass to OTXDataset
        """
        kwargs["sample_type"] = ClassificationSample
        super().__init__(**kwargs)

# class OTXMultilabelClsDataset(OTXDataset):
#     """OTXDataset class for multi-label classification task using new Datumaro experimental Dataset."""
#
#     def __init__(self, **kwargs) -> None:
#         """Initialize OTXMultilabelClsDataset.
#
#         Args:
#             **kwargs: Keyword arguments to pass to OTXDataset
#         """
#         kwargs["sample_type"] = MultiLabelClassificationSample
#         super().__init__(**kwargs)
#         self.num_classes = len(self.dm_subset.categories()[AnnotationType.label])
#
#     def _get_item_impl(self, index: int) -> MultiLabelClassificationSample | None:
#         item = self.dm_subset[index]
#         img = item.media_as(Image)
#         ignored_labels: list[int] = []  # This should be assigned form item
#         img_data, img_shape, _ = self._get_img_data_and_shape(img)
#
#         label_ids = set()
#         for ann in item.annotations:
#             # multilabel information stored in 'multi_label_ids' attribute when the source format is arrow
#             if "multi_label_ids" in ann.attributes:
#                 for lbl_idx in ann.attributes["multi_label_ids"]:
#                     label_ids.add(lbl_idx)
#
#             if isinstance(ann, Label):
#                 label_ids.add(ann.label)
#             else:
#                 # If the annotation is not Label, it should be converted to Label.
#                 # For Chained Task: Detection (Bbox) -> Classification (Label)
#                 label = Label(label=ann.label)
#                 label_ids.add(label.label)
#         labels = np.array(list(label_ids), dtype=np.int64)
#
#         image_info = ImageInfo(
#             width=img_data.shape[1],
#             height=img_data.shape[0],
#         )
#         # Create multilabel classification sample
#         sample = MultiLabelClassificationSample(
#             image=img_data,
#             labels=self._convert_to_onehot(labels, ignored_labels),
#             image_info=image_info,
#         )
#
#         return self._apply_transforms(sample)
#
#     def _convert_to_onehot(self, labels: np.ndarray, ignored_labels: list[int]) -> np.ndarray:
#         """Convert label to one-hot vector format."""
#         # Convert to torch tensor for one_hot
#         labels_tensor = torch.from_numpy(labels).long()
#         # Torch's one_hot() expects the input to be of type long
#         onehot = functional.one_hot(labels_tensor, self.num_classes).sum(0).clamp_max_(1).numpy()
#         if ignored_labels:
#             for ignore_label in ignored_labels:
#                 onehot[ignore_label] = -1
#         return onehot


# class OTXHlabelClsDataset(OTXDataset):
#     """OTXDataset class for H-label classification task using new Datumaro experimental Dataset."""
#
#     def __init__(self, **kwargs) -> None:
#         """Initialize OTXHlabelClsDataset.
#
#         Args:
#             **kwargs: Keyword arguments to pass to OTXDataset
#         """
#         # Set the sample type to HierarchicalClassificationSample
#         kwargs["sample_type"] = HierarchicalClassificationSample
#         super().__init__(**kwargs)
#         self.dm_categories = self.dm_subset.categories()[AnnotationType.label]
#
#         # Hlabel classification used HLabelInfo to insert the HLabelData.
#         if self.data_format == "arrow":
#             # arrow format stores label IDs as names, have to deal with that here
#             self.label_info = HLabelInfo.from_dm_label_groups_arrow(self.dm_categories)
#         else:
#             self.label_info = HLabelInfo.from_dm_label_groups(self.dm_categories)
#
#         self.id_to_name_mapping = dict(zip(self.label_info.label_ids, self.label_info.label_names))
#         self.id_to_name_mapping[""] = ""
#
#         if self.label_info.num_multiclass_heads == 0:
#             msg = "The number of multiclass heads should be larger than 0."
#             raise ValueError(msg)
#
#         if self.data_format != "arrow":
#             for dm_item in self.dm_subset:
#                 self._add_ancestors(dm_item.annotations)
#
#     def _add_ancestors(self, label_anns: list[Label]) -> None:
#         """Add ancestors recursively if some label miss the ancestor information.
#
#         If the label tree likes below,
#         object - vehicle -- car
#                          |- bus
#                          |- truck
#         And annotation = ['car'], it should be ['car', 'vehicle', 'object'], to include the ancestor.
#
#         This function add the ancestors to the annotation if missing.
#         """
#
#         def _label_idx_to_name(idx: int) -> str:
#             return self.dm_categories[idx].name
#
#         def _label_name_to_idx(name: str) -> int:
#             indices = [idx for idx, val in enumerate(self.label_info.label_names) if val == name]
#             return indices[0]
#
#         def _get_label_group_idx(label_name: str) -> int:
#             if isinstance(self.label_info, HLabelInfo):
#                 if self.data_format == "arrow":
#                     return self.label_info.class_to_group_idx[self.id_to_name_mapping[label_name]][0]
#                 return self.label_info.class_to_group_idx[label_name][0]
#             msg = f"self.label_info should have HLabelInfo type, got {type(self.label_info)}"
#             raise ValueError(msg)
#
#         def _find_ancestor_recursively(label_name: str, ancestors: list) -> list[str]:
#             _, dm_label_category = self.dm_categories.find(label_name)
#             parent_name = dm_label_category.parent if dm_label_category else ""
#
#             if parent_name != "":
#                 ancestors.append(parent_name)
#                 _find_ancestor_recursively(parent_name, ancestors)
#             return ancestors
#
#         def _get_all_label_names_in_anns(anns: list[Label]) -> list[str]:
#             return [_label_idx_to_name(ann.label) for ann in anns]
#
#         all_label_names = _get_all_label_names_in_anns(label_anns)
#         ancestor_dm_labels = []
#         for ann in label_anns:
#             label_idx = ann.label
#             label_name = _label_idx_to_name(label_idx)
#             ancestors = _find_ancestor_recursively(label_name, [])
#
#             for i, ancestor in enumerate(ancestors):
#                 if ancestor not in all_label_names:
#                     ancestor_dm_labels.append(
#                         Label(
#                             label=_label_name_to_idx(ancestor),
#                             id=len(label_anns) + i,
#                             group=_get_label_group_idx(ancestor),
#                         ),
#                     )
#         label_anns.extend(ancestor_dm_labels)
#
#     def _get_item_impl(self, index: int) -> HierarchicalClassificationSample | None:
#         item = self.dm_subset[index]
#         img = item.media_as(Image)
#         ignored_labels: list[int] = []  # This should be assigned form item
#         img_data, img_shape, _ = self._get_img_data_and_shape(img)
#
#         label_ids = set()
#         for ann in item.annotations:
#             # in h-cls scenario multilabel information stored in 'multi_label_ids' attribute
#             if "multi_label_ids" in ann.attributes:
#                 for lbl_idx in ann.attributes["multi_label_ids"]:
#                     label_ids.add(lbl_idx)
#
#             if isinstance(ann, Label):
#                 label_ids.add(ann.label)
#             else:
#                 # If the annotation is not Label, it should be converted to Label.
#                 # For Chained Task: Detection (Bbox) -> Classification (Label)
#                 label = Label(label=ann.label)
#                 label_ids.add(label.label)
#
#         hlabel_labels = self._convert_label_to_hlabel_format([Label(label=idx) for idx in label_ids], ignored_labels)
#
#         # Create image info sample
#         image_info = ImageInfo(
#             width=img_data.shape[1],
#             height=img_data.shape[0],
#         )
#
#         # Create hierarchical classification sample
#         sample = HierarchicalClassificationSample(
#             image=img_data,
#             labels=np.array(hlabel_labels, dtype=np.int64),
#             image_info=image_info,
#         )
#
#         return self._apply_transforms(sample)
#
#     def _convert_label_to_hlabel_format(self, label_anns: list[Label], ignored_labels: list[int]) -> list[int]:
#         """Convert format of the label to the h-label.
#
#         It converts the label format to h-label format.
#         Total length of result is sum of number of hierarchy and number of multilabel classes.
#
#         i.e.
#         Let's assume that we used the same dataset with example of the definition of HLabelData
#         and the original labels are ["Rigid", "Triangle", "Lion"].
#
#         Then, h-label format will be [0, 1, 1, 0].
#         The first N-th indices represent the label index of multiclass heads (N=num_multiclass_heads),
#         others represent the multilabel labels.
#
#         [Multiclass Heads]
#         0-th index = 0 -> ["Rigid"(O), "Non-Rigid"(X)] <- First multiclass head
#         1-st index = 1 -> ["Rectangle"(O), "Triangle"(X), "Circle"(X)] <- Second multiclass head
#
#         [Multilabel Head]
#         2, 3 indices = [1, 0] -> ["Lion"(O), "Panda"(X)]
#         """
#         if not isinstance(self.label_info, HLabelInfo):
#             msg = f"The type of label_info should be HLabelInfo, got {type(self.label_info)}."
#             raise TypeError(msg)
#
#         num_multiclass_heads = self.label_info.num_multiclass_heads
#         num_multilabel_classes = self.label_info.num_multilabel_classes
#
#         class_indices = [0] * (num_multiclass_heads + num_multilabel_classes)
#         for i in range(num_multiclass_heads):
#             class_indices[i] = -1
#
#         for ann in label_anns:
#             if self.data_format == "arrow":
#                 # skips unknown labels for instance, the empty one
#                 if self.dm_categories.items[ann.label].name not in self.id_to_name_mapping:
#                     continue
#                 ann_name = self.id_to_name_mapping[self.dm_categories.items[ann.label].name]
#             else:
#                 ann_name = self.dm_categories.items[ann.label].name
#             group_idx, in_group_idx = self.label_info.class_to_group_idx[ann_name]
#
#             if group_idx < num_multiclass_heads:
#                 class_indices[group_idx] = in_group_idx
#             elif ann.label not in ignored_labels:
#                 class_indices[num_multiclass_heads + in_group_idx] = 1
#             else:
#                 class_indices[num_multiclass_heads + in_group_idx] = -1
#
#         return class_indices