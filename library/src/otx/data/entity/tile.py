# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTX tile data entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
from torchvision import tv_tensors

from otx.data.entity.sample import OTXSample
from otx.data.entity.torch import OTXDataBatch
from otx.data.entity.utils import stack_batch
from otx.types.task import OTXTaskType

from .base import ImageInfo

if TYPE_CHECKING:
    import numpy as np
    from datumaro.experimental.fields import TileInfo
    from torch import LongTensor


@dataclass
class TileDataEntity:
    """Base data entity for tile task.

    Attributes:
        num_tiles (int): The number of tiles.
        entity_list (Sequence[OTXDataEntity]): A list of OTXDataEntity.
        ori_img_info (ImageInfo): The image information about the original image.
    """

    num_tiles: int
    entity_list: Sequence[OTXSample]
    ori_img_info: ImageInfo

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        raise NotImplementedError


@dataclass
class TileDetDataEntity(TileDataEntity):
    """Data entity for detection tile task.

    Attributes:
        ori_bboxes (tv_tensors.BoundingBoxes): The bounding boxes of the original image.
        ori_labels (LongTensor): The labels of the original image.
    """

    ori_bboxes: tv_tensors.BoundingBoxes
    ori_labels: LongTensor

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.DETECTION


TileAttrDictList = list[dict[str, int | str]]


@dataclass
class OTXTileBatchDataEntity:
    """Base batch data entity for tile task.

    Attributes:
        batch_size (int): The size of the batch.
        batch_tiles (list[list[tv_tensors.Image]]): The batch of tile images.
        batch_tile_img_infos (list[list[ImageInfo]]): The batch of tiles image information.
        imgs_info (list[ImageInfo]): The image information about the original image.
    """

    batch_size: int
    batch_tiles: list[list[tv_tensors.Image]]
    batch_tile_img_infos: list[list[ImageInfo]]
    batch_tile_tile_infos: list[list[TileInfo]]
    imgs_info: list[ImageInfo]

    def unbind(self) -> list[tuple[list[TileInfo], OTXDataBatch]]:
        """Unbind batch data entity."""
        raise NotImplementedError


@dataclass
class TileBatchDetDataEntity(OTXTileBatchDataEntity):
    """Batch data entity for detection tile task.

    Attributes:
        bboxes (list[tv_tensors.BoundingBoxes]): The bounding boxes of the original image.
        labels (list[LongTensor]): The labels of the original image.
    """

    bboxes: list[tv_tensors.BoundingBoxes]
    labels: list[LongTensor]

    def unbind(self) -> list[tuple[list[TileInfo], OTXDataBatch]]:
        """Unbind batch data entity for detection task."""
        tiles = [tile for tiles in self.batch_tiles for tile in tiles]
        tile_img_infos = [tile_info for tile_infos in self.batch_tile_img_infos for tile_info in tile_infos]
        tile_tile_infos = [tile_info for tile_infos in self.batch_tile_tile_infos for tile_info in tile_infos]

        batch_data_entities = []
        for i in range(0, len(tiles), self.batch_size):
            stacked_images, updated_img_info = stack_batch(
                tiles[i : i + self.batch_size],
                tile_img_infos[i : i + self.batch_size],
            )
            batch_data_entities.append(
                (
                    tile_tile_infos[i : i + self.batch_size],
                    OTXDataBatch(
                        batch_size=self.batch_size,
                        images=stacked_images,
                        imgs_info=updated_img_info,
                    ),
                )
            )
        return batch_data_entities

    @classmethod
    def collate_fn(cls, batch_entities: list[TileDetDataEntity]) -> TileBatchDetDataEntity:
        """Collate function to collect TileDetDataEntity into TileBatchDetDataEntity in data loader."""
        if (batch_size := len(batch_entities)) == 0:
            msg = "collate_fn() input should have > 0 entities"
            raise RuntimeError(msg)

        for tile_entity in batch_entities:
            for entity in tile_entity.entity_list:
                if not isinstance(entity, OTXSample):
                    msg = "All entities should be OTXSample before collate_fn()"
                    raise TypeError(msg)
                if entity.img_info is None:
                    msg = "All entities should have img_info, but found None"
                    raise ValueError(msg)

        return TileBatchDetDataEntity(
            batch_size=batch_size,
            batch_tiles=[[entity.image for entity in tile_entity.entity_list] for tile_entity in batch_entities],
            batch_tile_img_infos=[
                [entity.img_info for entity in tile_entity.entity_list] for tile_entity in batch_entities
            ],
            batch_tile_tile_infos=[
                [entity.tile for entity in tile_entity.entity_list] for tile_entity in batch_entities
            ],
            imgs_info=[tile_entity.ori_img_info for tile_entity in batch_entities],
            bboxes=[tile_entity.ori_bboxes for tile_entity in batch_entities],
            labels=[tile_entity.ori_labels for tile_entity in batch_entities],
        )


@dataclass
class TileInstSegDataEntity(TileDataEntity):
    """Data entity for instance segmentation tile task.

    Attributes:
        ori_bboxes (tv_tensors.BoundingBoxes): The bounding boxes of the original image.
        ori_labels (LongTensor): The labels of the original image.
        ori_masks (tv_tensors.Mask): The masks of the original image.
        ori_polygons (list[np.ndarray]): The polygons of the original image as arrays shaped (K, 2).
    """

    ori_bboxes: tv_tensors.BoundingBoxes
    ori_labels: LongTensor
    ori_masks: tv_tensors.Mask
    ori_polygons: list[np.ndarray]

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.INSTANCE_SEGMENTATION


@dataclass
class TileBatchInstSegDataEntity(OTXTileBatchDataEntity):
    """Batch data entity for instance segmentation tile task.

    Attributes:
        bboxes (list[tv_tensors.BoundingBoxes]): The bounding boxes of the original image.
        labels (list[LongTensor]): The labels of the original image.
        masks (list[tv_tensors.Mask]): The masks of the original image.
        polygons (list[list[np.ndarray]]): The polygons of the original image as arrays shaped (K, 2).
    """

    bboxes: list[tv_tensors.BoundingBoxes]
    labels: list[LongTensor]
    masks: list[tv_tensors.Mask]
    polygons: list[list[np.ndarray]]

    def unbind(self) -> list[tuple[TileAttrDictList, OTXDataBatch]]:
        """Unbind batch data entity for instance segmentation task."""
        tiles = [tile for tiles in self.batch_tiles for tile in tiles]
        tile_img_infos = [tile_info for tile_infos in self.batch_tile_img_infos for tile_info in tile_infos]
        tile_tile_infos = [tile_info for tile_infos in self.batch_tile_tile_infos for tile_info in tile_infos]

        batch_data_entities = [
            (
                tile_tile_infos[i : i + self.batch_size],
                OTXDataBatch(
                    batch_size=self.batch_size,
                    images=tiles[i : i + self.batch_size],
                    imgs_info=tile_img_infos[i : i + self.batch_size],
                ),
            )
            for i in range(0, len(tiles), self.batch_size)
        ]
        return list(batch_data_entities)

    @classmethod
    def collate_fn(cls, batch_entities: list[TileInstSegDataEntity]) -> TileBatchInstSegDataEntity:
        """Collate function to collect TileInstSegDataEntity into TileBatchInstSegDataEntity in data loader."""
        if (batch_size := len(batch_entities)) == 0:
            msg = "collate_fn() input should have > 0 entities"
            raise RuntimeError(msg)

        for tile_entity in batch_entities:
            for entity in tile_entity.entity_list:
                if not isinstance(entity, OTXSample):
                    msg = "All entities should be OTXSample before collate_fn()"
                    raise TypeError(msg)
                if entity.img_info is None:
                    msg = "All entities should have img_info, but found None"
                    raise ValueError(msg)

        return TileBatchInstSegDataEntity(
            batch_size=batch_size,
            batch_tiles=[[entity.image for entity in tile_entity.entity_list] for tile_entity in batch_entities],
            batch_tile_img_infos=[
                [entity.img_info for entity in tile_entity.entity_list if isinstance(entity.img_info, ImageInfo)]
                for tile_entity in batch_entities
            ],
            batch_tile_tile_infos=[
                [entity.tile for entity in tile_entity.entity_list] for tile_entity in batch_entities
            ],
            imgs_info=[tile_entity.ori_img_info for tile_entity in batch_entities],
            bboxes=[tile_entity.ori_bboxes for tile_entity in batch_entities],
            labels=[tile_entity.ori_labels for tile_entity in batch_entities],
            masks=[tile_entity.ori_masks for tile_entity in batch_entities],
            polygons=[tile_entity.ori_polygons for tile_entity in batch_entities],
        )


@dataclass
class TileSegDataEntity(TileDataEntity):
    """Data entity for segmentation tile task.

    Attributes:
        ori_masks (tv_tensors.Mask): The masks of the original image.
    """

    ori_masks: tv_tensors.Mask

    @property
    def task(self) -> OTXTaskType:
        """OTX Task type definition."""
        return OTXTaskType.SEMANTIC_SEGMENTATION


@dataclass
class TileBatchSegDataEntity(OTXTileBatchDataEntity):
    """Batch data entity for semantic segmentation tile task.

    Attributes:
        masks (list[tv_tensors.Mask]): The masks of the original image.
    """

    masks: list[tv_tensors.Mask]

    def unbind(self) -> list[tuple[list[dict[str, int | str]], OTXDataBatch]]:
        """Unbind batch data entity for semantic segmentation task."""
        tiles = [tile for tiles in self.batch_tiles for tile in tiles]
        tile_img_infos = [tile_info for tile_infos in self.batch_tile_img_infos for tile_info in tile_infos]
        tile_tile_infos = [tile_info for tile_infos in self.batch_tile_tile_infos for tile_info in tile_infos]

        batch_data_entities = [
            (
                tile_tile_infos[i : i + self.batch_size],
                OTXDataBatch(
                    batch_size=self.batch_size,
                    images=tv_tensors.wrap(torch.stack(tiles[i : i + self.batch_size]), like=tiles[0]),
                    imgs_info=tile_img_infos[i : i + self.batch_size],
                    masks=[torch.empty((1, 1, 1)) for _ in range(self.batch_size)],
                ),
            )
            for i in range(0, len(tiles), self.batch_size)
        ]
        return list(batch_data_entities)

    @classmethod
    def collate_fn(cls, batch_entities: list[TileSegDataEntity]) -> TileBatchSegDataEntity:
        """Collate function to collect TileSegDataEntity into TileBatchSegDataEntity in data loader."""
        if (batch_size := len(batch_entities)) == 0:
            msg = "collate_fn() input should have > 0 entities"
            raise RuntimeError(msg)

        for tile_entity in batch_entities:
            for entity in tile_entity.entity_list:
                if not isinstance(entity, OTXSample):
                    msg = "All entities should be OTXSample before collate_fn()"
                    raise TypeError(msg)
                if entity.img_info is None:
                    msg = "All entities should have img_info, but found None"
                    raise ValueError(msg)

        return TileBatchSegDataEntity(
            batch_size=batch_size,
            batch_tiles=[[entity.image for entity in tile_entity.entity_list] for tile_entity in batch_entities],
            batch_tile_img_infos=[
                [entity.img_info for entity in tile_entity.entity_list] for tile_entity in batch_entities
            ],
            batch_tile_tile_infos=[
                [entity.tile for entity in tile_entity.entity_list] for tile_entity in batch_entities
            ],
            imgs_info=[tile_entity.ori_img_info for tile_entity in batch_entities],
            masks=[tile_entity.ori_masks for tile_entity in batch_entities],
        )
