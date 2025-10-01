# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX tile dataset."""

from __future__ import annotations

import logging as log
import operator
import warnings
from collections import defaultdict
from copy import deepcopy
from itertools import product
from typing import TYPE_CHECKING, Callable

import numpy as np
import shapely.geometry as sg
import torch
from datumaro.components.annotation import Bbox, Ellipse, Polygon
from datumaro.experimental.dataset import Sample, Dataset as DmDataset
from datumaro.experimental.schema import Schema
from datumaro.experimental.tiling.tiler_registry import create_tiling_transform, TilingConfig
from torchvision import tv_tensors

from otx.data.dataset.segmentation import _extract_class_mask
from otx.data.entity.base import ImageInfo
from otx.data.entity.tile import (
    TileBatchDetDataEntity,
    TileBatchInstSegDataEntity,
    TileBatchSegDataEntity,
    TileDetDataEntity,
    TileInstSegDataEntity,
    TileSegDataEntity,
)
from otx.data.entity.torch import OTXDataItem
from otx.data.utils.structures.mask.mask_util import polygon_to_bitmap
from otx.types.task import OTXTaskType

from .base_new import OTXDataset

if TYPE_CHECKING:
    from datumaro.components.media import BboxIntCoords

    from otx.config.data import TileConfig
    from otx.data.dataset.detection import OTXDetectionDataset
    from otx.data.dataset.instance_segmentation import OTXInstanceSegDataset
    from otx.data.dataset.segmentation import OTXSegmentationDataset

# ruff: noqa: SLF001
# NOTE: Disable private-member-access (SLF001).
# This is a workaround so we could apply the same transforms to tiles as the original dataset.

# NOTE: Datumaro subset name should be standardized.
TRAIN_SUBSET_NAMES = ("train", "TRAINING")
VAL_SUBSET_NAMES = ("val", "VALIDATION")


class OTXTileDatasetFactory:
    """OTX tile dataset factory."""

    @classmethod
    def create(
        cls,
        task: OTXTaskType,
        dataset: OTXDataset,
        tile_config: TileConfig,
    ) -> OTXDataset:
        """Create a tile dataset based on the task type and subset type.

        NOte: All task utilize the same OTXTileTrainDataset for training.
              In testing, we use different tile dataset for different task
              type due to different annotation format and data entity.

        Args:
            task (OTXTaskType): OTX task type.
            dataset (OTXDataset): OTX dataset.
            tile_config (TilerConfig): Tile configuration.

        Returns:
            OTXTileDataset: Tile dataset.
        """
        if dataset.dm_subset[0].subset in TRAIN_SUBSET_NAMES:
            dm_dataset = dataset.dm_subset
            dm_dataset = dm_dataset.transform(create_tiling_transform(
                TilingConfig(
                    tile_height=tile_config.tile_size[0],
                    tile_width=tile_config.tile_size[1],
                    overlap_x=tile_config.overlap,
                    overlap_y=tile_config.overlap,
                ),
                threshold_drop_ann=0.5,
            ))
            # TODO(gdlg): fixme
            # dm_dataset = dm_dataset.filter("/item/annotation", filter_annotations=True, remove_empty=True)
            dataset.dm_subset = dm_dataset
            return dataset

        if task == OTXTaskType.DETECTION:
            return OTXTileDetTestDataset(dataset, tile_config)
        if task in [OTXTaskType.ROTATED_DETECTION, OTXTaskType.INSTANCE_SEGMENTATION]:
            return OTXTileInstSegTestDataset(dataset, tile_config)
        if task == OTXTaskType.SEMANTIC_SEGMENTATION:
            return OTXTileSemanticSegTestDataset(dataset, tile_config)

        msg = f"Unsupported task type: {task} for tiling"
        raise NotImplementedError(msg)


class OTXTileDataset(OTXDataset):
    """OTX tile dataset base class.

    Args:
        dataset (OTXDataset): OTX dataset.
        tile_config (TilerConfig): Tile configuration.
    """

    def __init__(self, dataset: OTXDataset, tile_config: TileConfig) -> None:
        super().__init__(
            dataset.dm_subset,
            dataset.transforms,
            dataset.max_refetch,
            dataset.image_color_channel,
            dataset.stack_images,
            dataset.to_tv_image,
        )
        self.tile_config = tile_config
        self._dataset = dataset

        # LabelInfo differs from SegLabelInfo, thus we need to update it for semantic segmentation.
        if self.label_info != dataset.label_info:
            msg = (
                "Replace the label info to match the dataset's label info",
                "as there is a mismatch between the dataset and the tile dataset.",
            )
            log.warning(msg)
            self.label_info = dataset.label_info

    def __len__(self) -> int:
        return len(self._dataset)


class OTXTileDatasetWithGetTiles(OTXDataset):
    def __init__(self, dataset: OTXDataset, tile_config: TileConfig) -> None:
        super().__init__(dataset, tile_config)

    def transform_item(
        self,
        parent_idx: int,
        tile_size: tuple[int, int],
        overlap: tuple[float, float],
        with_full_img: bool,
    ) -> DmDataset[Sample]:
        """Transform a dataset item to tile dataset which contains multiple tiles."""
        self.
        tile_ds = DmDataset.from_iterable([item])
        return tile_ds.transform(
            OTXTileTransform,
            tile_size=tile_size,
            overlap=overlap,
            threshold_drop_ann=0.5,
            with_full_img=with_full_img,
        )

    @property
    def collate_fn(self) -> Callable:
        """Collate function from the original dataset."""
        return self._dataset.collate_fn

    def _convert_entity(self, image: np.ndarray, dataset_item: DatasetItem, parent_idx: int) -> OTXDataItem:
        """Convert a tile dataset item to OTXDataItem."""
        msg = "Method _convert_entity is not implemented."
        raise NotImplementedError(msg)

    def _get_item_impl(self, index: int) -> OTXDataItem | None:
        """Get item implementation from the original dataset."""
        return self._dataset._get_item_impl(index)

    def get_tiles(
        self,
        parent_idx: int,
    ) -> tuple[list[OTXDataItem], list[dict]]:
        """Retrieves tiles from the given image and dataset item.

        Args:
            image (np.ndarray): The input image.
            item (DatasetItem): The dataset item.
            parent_idx (int): The parent index. This is to keep track of the original dataset item index for merging.

        Returns:
            A tuple containing two lists:
            - tile_entities (list[OTXDataItem]): List of tile entities.
            - tile_attrs (list[dict]): List of tile attributes.
        """
        tile_ds = self.transform_item(
            parent_idx,
            tile_size=self.tile_config.tile_size,
            overlap=(self.tile_config.overlap, self.tile_config.overlap),
            with_full_img=self.tile_config.with_full_img,
        )

        if item.subset in VAL_SUBSET_NAMES:
            # NOTE: filter validation tiles with annotations only to avoid evaluation on empty tiles.
            tile_ds = tile_ds.filter("/item/annotation", filter_annotations=True, remove_empty=True)
            # if tile dataset is empty it means objects are too big to fit in any tile, in this case include full image
            if len(tile_ds) == 0:
                tile_ds = self.transform_item(
                    item,
                    tile_size=self.tile_config.tile_size,
                    overlap=(self.tile_config.overlap, self.tile_config.overlap),
                    with_full_img=True,
                )

        tile_entities: list[OTXDataItem] = []
        tile_attrs: list[dict] = []
        for tile in tile_ds:
            tile_entity = self._convert_entity(image, tile, parent_idx)
            # apply the same transforms as the original dataset
            transformed_tile = self._apply_transforms(tile_entity)
            if transformed_tile is None:
                msg = "Transformed tile is None"
                raise RuntimeError(msg)
            tile.attributes.update({"tile_size": self.tile_config.tile_size})
            tile_entities.append(transformed_tile)
            tile_attrs.append(tile.attributes)
        return tile_entities, tile_attrs


class OTXTileTrainDataset(OTXTileDataset):
    """OTX tile train dataset.

    Args:
        dataset (OTXDataset): OTX dataset.
        tile_config (TilerConfig): Tile configuration.
    """

    def __init__(self, dataset: OTXDataset, tile_config: TileConfig) -> None:
        dm_dataset = dataset.dm_subset
        dm_dataset = dm_dataset.transform(create_tiling_transform(
            TilingConfig(
                tile_height=tile_config.tile_size[0],
                tile_width=tile_config.tile_size[1],
                overlap_x=tile_config.overlap,
                overlap_y=tile_config.overlap,
            ),
            threshold_drop_ann=0.5,
        ))
        # TODO(gdlg): fixme
        # dm_dataset = dm_dataset.filter("/item/annotation", filter_annotations=True, remove_empty=True)
        # Include original dataset for training

        dataset.dm_subset = dm_dataset
        super().__init__(dataset, tile_config)


class OTXTileDetTestDataset(OTXTileDatasetWithGetTiles):
    """OTX tile detection test dataset.

    OTXTileDetTestDataset wraps a list of tiles (DetDataEntity) into a single TileDetDataEntity for testing/predicting.

    Args:
        dataset (OTXDetDataset): OTX detection dataset.
        tile_config (TilerConfig): Tile configuration.
    """

    def __init__(self, dataset: OTXDetectionDataset, tile_config: TileConfig) -> None:
        super().__init__(dataset, tile_config)

    @property
    def collate_fn(self) -> Callable:
        """Collate function for tile detection test dataset."""
        return TileBatchDetDataEntity.collate_fn

    def _get_item_impl(self, index: int) -> TileDetDataEntity:  # type: ignore[override]
        """Get item implementation.

        Transform a single dataset item to multiple tiles using Datumaro tiling plugin, and
        wrap tiles into a single TileDetDataEntity.

        Args:
            index (int): Index of the dataset item.

        Returns:
            TileDetDataEntity: tile detection data entity that wraps a list of detection data entities.

        Note:
            Ignoring [override] check is necessary here since OTXDataset._get_item_impl exclusively permits
            the return of OTXDataItem. Nevertheless, in instances involving tiling, it becomes
            imperative to encapsulate tiles within a unified entity, namely TileDetDataEntity.
        """
        item = self.dm_subset[index]
        img = item.media_as(Image)
        img_data, img_shape, _ = self._get_img_data_and_shape(img)

        bbox_anns = [ann for ann in item.annotations if isinstance(ann, Bbox)]

        bboxes = (
            np.stack([ann.points for ann in bbox_anns], axis=0).astype(np.float32)
            if len(bbox_anns) > 0
            else np.zeros((0, 4), dtype=np.float32)
        )
        labels = torch.as_tensor([ann.label for ann in bbox_anns])

        tile_entities, tile_attrs = self.get_tiles(index)

        return TileDetDataEntity(
            num_tiles=len(tile_entities),
            entity_list=tile_entities,
            tile_attr_list=tile_attrs,
            ori_img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
            ),
            ori_bboxes=tv_tensors.BoundingBoxes(
                bboxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=img_shape,
            ),
            ori_labels=labels,
        )

    def _convert_entity(self, image: np.ndarray, dataset_item: DatasetItem, parent_idx: int) -> OTXDataItem:  # type: ignore[override]
        """Convert a tile datumaro dataset item to TorchDataItem."""
        x1, y1, w, h = dataset_item.attributes["roi"]
        tile_img = image[y1 : y1 + h, x1 : x1 + w]
        tile_shape = tile_img.shape[:2]
        img_info = ImageInfo(
            img_idx=parent_idx,
            img_shape=tile_shape,
            ori_shape=tile_shape,
        )
        return OTXDataItem(
            image=tile_img,
            img_info=img_info,
        )


class OTXTileInstSegTestDataset(OTXTileDataset):
    """OTX tile inst-seg test dataset.

    OTXTileDetTestDataset wraps a list of tiles (TorchDataItem) into a single TileDetDataEntity
    for testing/predicting.

    Args:
        dataset (OTXInstanceSegDataset): OTX inst-seg dataset.
        tile_config (TilerConfig): Tile configuration.
    """

    def __init__(self, dataset: OTXInstanceSegDataset, tile_config: TileConfig) -> None:
        super().__init__(dataset, tile_config)

    @property
    def collate_fn(self) -> Callable:
        """Collate function for tile inst-seg test dataset."""
        return TileBatchInstSegDataEntity.collate_fn

    def _get_item_impl(self, index: int) -> TileInstSegDataEntity:  # type: ignore[override]
        """Get item implementation.

        Transform a single dataset item to multiple tiles using Datumaro tiling plugin, and
        wrap tiles into a single TileInstSegDataEntity.

        Args:
            index (int): Index of the dataset item.

        Returns:
            TileInstSegDataEntity: tile inst-seg data entity that wraps a list of inst-seg data entities.

        Note:
            Ignoring [override] check is necessary here since OTXDataset._get_item_impl exclusively permits
            the return of OTXDataItem. Nevertheless, in instances involving tiling, it becomes
            imperative to encapsulate tiles within a unified entity, namely TileInstSegDataEntity.
        """
        item = self.dm_subset[index]
        img = item.media_as(Image)
        img_data, img_shape, _ = self._get_img_data_and_shape(img)

        anno_collection: dict[str, list] = defaultdict(list)
        for anno in item.annotations:
            anno_collection[anno.__class__.__name__].append(anno)

        gt_bboxes, gt_labels, gt_masks, gt_polygons = [], [], [], []

        # TODO(Eugene): https://jira.devtools.intel.com/browse/CVS-159363
        # Temporary solution to handle multiple annotation types.
        # Ideally, we should pre-filter annotations during initialization of the dataset.

        if Polygon.__name__ in anno_collection:  # Polygon for InstSeg has higher priority
            for poly in anno_collection[Polygon.__name__]:
                bbox = Bbox(*poly.get_bbox()).points
                gt_bboxes.append(bbox)
                gt_labels.append(poly.label)

                if self._dataset.include_polygons:
                    gt_polygons.append(poly)
                else:
                    gt_masks.append(polygon_to_bitmap([poly], *img_shape)[0])
        elif Bbox.__name__ in anno_collection:
            boxes = anno_collection[Bbox.__name__]
            gt_bboxes = [ann.points for ann in boxes]
            gt_labels = [ann.label for ann in boxes]
            for box in boxes:
                poly = Polygon(box.as_polygon())
                if self._dataset.include_polygons:
                    gt_polygons.append(poly)
                else:
                    gt_masks.append(polygon_to_bitmap([poly], *img_shape)[0])
        elif Ellipse.__name__ in anno_collection:
            for ellipse in anno_collection[Ellipse.__name__]:
                bbox = Bbox(*ellipse.get_bbox()).points
                gt_bboxes.append(bbox)
                gt_labels.append(ellipse.label)
                poly = Polygon(ellipse.as_polygon(num_points=10))
                if self._dataset.include_polygons:
                    gt_polygons.append(poly)
                else:
                    gt_masks.append(polygon_to_bitmap([poly], *img_shape)[0])
        else:
            warnings.warn(f"No valid annotations found for image {item.id}!", stacklevel=2)

        bboxes = np.stack(gt_bboxes, dtype=np.float32) if gt_bboxes else np.empty((0, 4), dtype=np.float32)
        masks = np.stack(gt_masks, axis=0) if gt_masks else np.empty((0, *img_shape), dtype=bool)
        labels = np.array(gt_labels, dtype=np.int64)

        tile_entities, tile_attrs = self.get_tiles(img_data, item, index)

        return TileInstSegDataEntity(
            num_tiles=len(tile_entities),
            entity_list=tile_entities,
            tile_attr_list=tile_attrs,
            ori_img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
            ),
            ori_bboxes=tv_tensors.BoundingBoxes(
                bboxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=img_shape,
            ),
            ori_labels=torch.as_tensor(labels),
            ori_masks=tv_tensors.Mask(masks, dtype=torch.uint8),
            ori_polygons=gt_polygons,
        )

    def _convert_entity(self, image: np.ndarray, dataset_item: DatasetItem, parent_idx: int) -> OTXDataItem:  # type: ignore[override]
        """Convert a tile dataset item to TorchDataItem."""
        x1, y1, w, h = dataset_item.attributes["roi"]
        tile_img = image[y1 : y1 + h, x1 : x1 + w]
        tile_shape = tile_img.shape[:2]
        img_info = ImageInfo(
            img_idx=parent_idx,
            img_shape=tile_shape,
            ori_shape=tile_shape,
        )
        return OTXDataItem(
            image=tile_img,
            img_info=img_info,
            masks=tv_tensors.Mask(np.zeros((0, *tile_shape), dtype=bool)),
        )


class OTXTileSemanticSegTestDataset(OTXTileDataset):
    """OTX tile semantic-seg test dataset.

    OTXTileSemanticSegTestDataset wraps a list of tiles (SegDataEntity) into a single TileSegDataEntity
    for testing/predicting.

    Args:
        dataset (OTXSegmentationDataset): OTX semantic-seg dataset.
        tile_config (TilerConfig): Tile configuration.
    """

    def __init__(self, dataset: OTXSegmentationDataset, tile_config: TileConfig) -> None:
        super().__init__(dataset, tile_config)
        self.ignore_index = self._dataset.ignore_index

    @property
    def collate_fn(self) -> Callable:
        """Collate function for tile detection test dataset."""
        return TileBatchSegDataEntity.collate_fn

    def _get_item_impl(self, index: int) -> TileSegDataEntity:  # type: ignore[override]
        """Get item implementation.

        Transform a single dataset item to multiple tiles using Datumaro tiling plugin, and
        wrap tiles into a single TileSegDataEntity.

        Args:
            index (int): Index of the dataset item.

        Returns:
            TileSegDataEntity: tile semantic-seg data entity that wraps a list of semantic-seg data entities.
        """
        item = self.dm_subset[index]
        img = item.media_as(Image)
        img_data, img_shape, _ = self._get_img_data_and_shape(img)

        extracted_mask = _extract_class_mask(item=item, img_shape=img_shape, ignore_index=self.ignore_index)
        masks = tv_tensors.Mask(extracted_mask[None])
        tile_entities, tile_attrs = self.get_tiles(img_data, item, index)

        return TileSegDataEntity(
            num_tiles=len(tile_entities),
            entity_list=tile_entities,
            tile_attr_list=tile_attrs,
            ori_img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
            ),
            ori_masks=masks,
        )

    def _convert_entity(self, image: np.ndarray, dataset_item: DatasetItem, parent_idx: int) -> OTXDataItem:  # type: ignore[override]
        """Convert a tile datumaro dataset item to SegDataEntity."""
        x1, y1, w, h = dataset_item.attributes["roi"]
        tile_img = image[y1 : y1 + h, x1 : x1 + w]
        tile_shape = tile_img.shape[:2]
        img_info = ImageInfo(
            img_idx=parent_idx,
            img_shape=tile_shape,
            ori_shape=tile_shape,
        )
        return OTXDataItem(
            image=tile_img,
            img_info=img_info,
            masks=tv_tensors.Mask(np.zeros((0, *tile_shape), dtype=bool)),
        )
