"""Torch-specific tile data item implementations."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.utils import stack_batch

from .data import TorchDataBatch, TorchDataItem

if TYPE_CHECKING:
    import torch
    from torchvision.tv_tensors import BoundingBoxes, Image


@dataclass
class TorchTileDataItem:
    """Torch tile data item implementation."""

    num_tiles: int
    entity_list: list[TorchDataItem]
    tile_attr_list: list[dict[str, int | str]]
    ori_img_info: ImageInfo
    ori_bboxes: BoundingBoxes
    ori_labels: torch.LongTensor

    @staticmethod
    def collate_fn(batch_entities: list[TorchTileDataItem]) -> TorchTileDataBatch:
        """Collate function to collect TorchTileDataItem into TorchTileDataBatch in data loader."""
        if (batch_size := len(batch_entities)) == 0:
            msg = "collate_fn() input should have > 0 entities"
            raise RuntimeError(msg)

        return TorchTileDataBatch(
            batch_size=batch_size,
            batch_tiles=[[entity.image for entity in tile_entity.entity_list] for tile_entity in batch_entities],
            batch_tile_img_infos=[
                [entity.img_info for entity in tile_entity.entity_list]  # type: ignore[misc]
                for tile_entity in batch_entities
            ],
            batch_tile_attr_list=[tile_entity.tile_attr_list for tile_entity in batch_entities],  # type: ignore[misc]
            imgs_info=[tile_entity.ori_img_info for tile_entity in batch_entities],
            bboxes=[tile_entity.ori_bboxes for tile_entity in batch_entities],
            labels=[tile_entity.ori_labels for tile_entity in batch_entities],
        )


@dataclass
class TorchTileDataBatch:
    """Torch tile data batch implementation."""

    batch_size: int
    batch_tiles: list[list[Image]]
    batch_tile_img_infos: list[list[ImageInfo]]
    batch_tile_attr_list: list[dict[str, int | str]]
    imgs_info: list[ImageInfo]
    bboxes: list[BoundingBoxes]
    labels: list[torch.LongTensor]

    def unbind(self) -> list[tuple[dict[str, int | str], TorchDataBatch]]:
        """Unbind batch data."""
        tiles = [tile for tiles in self.batch_tiles for tile in tiles]
        tile_infos = [tile_info for tile_infos in self.batch_tile_img_infos for tile_info in tile_infos]
        tile_attr_list = [tile_attr for tile_attrs in self.batch_tile_attr_list for tile_attr in tile_attrs]

        batch_tile_attr_list = [
            tile_attr_list[i : i + self.batch_size] for i in range(0, len(tile_attr_list), self.batch_size)
        ]

        batch_data_entities = []
        for i in range(0, len(tiles), self.batch_size):
            stacked_images, updated_img_info = stack_batch(
                tiles[i : i + self.batch_size],
                tile_infos[i : i + self.batch_size],
            )
            batch_data_entities.append(
                TorchDataBatch(
                    batch_size=self.batch_size,
                    images=stacked_images,
                    imgs_info=updated_img_info,  # type: ignore[arg-type]
                    bboxes=None,
                    labels=None,
                ),
            )
        return list(zip(batch_tile_attr_list, batch_data_entities, strict=True))  # type: ignore[arg-type]
