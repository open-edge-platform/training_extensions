# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX tile dataset."""

from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING, Callable

from datumaro.experimental.fields import Subset
from datumaro.experimental.filtering.filter_registry import create_filtering_transform
from datumaro.experimental.tiling.tiler_registry import TilingConfig, create_tiling_transform

from getitune.data.entity.sample import OTXSample
from getitune.data.entity.tile import (
    TileBatchDetDataEntity,
    TileBatchInstSegDataEntity,
    TileBatchSegDataEntity,
    TileDetDataEntity,
    TileInstSegDataEntity,
    TileSegDataEntity,
)
from getitune.types.task import OTXTaskType

from .base import OTXDataset, _ensure_chw_format

if TYPE_CHECKING:
    from getitune.config.data import TileConfig
    from getitune.data.dataset.detection import OTXDetectionDataset
    from getitune.data.dataset.instance_segmentation import OTXInstanceSegDataset
    from getitune.data.dataset.segmentation import OTXSegmentationDataset

# ruff: noqa: SLF001
# NOTE: Disable private-member-access (SLF001).
# This is a workaround so we could apply the same transforms to tiles as the original dataset.


class OTXTileDatasetFactory:
    """OTX tile dataset factory."""

    @classmethod
    def create(
        cls,
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
        subset = dataset.dm_subset[0].subset  # type: ignore[attr-defined]
        if subset == Subset.TRAINING:
            dm_dataset = dataset.dm_subset
            dm_dataset = dm_dataset.transform(  # type: ignore[attr-defined]
                create_tiling_transform(
                    TilingConfig(
                        tile_height=tile_config.tile_size[0],
                        tile_width=tile_config.tile_size[1],
                        overlap_x=tile_config.overlap,
                        overlap_y=tile_config.overlap,
                    ),
                    threshold_drop_ann=0.5,
                ),
                dtype=dm_dataset.dtype,  # type: ignore[attr-defined]
            )
            dm_dataset = dm_dataset.transform(create_filtering_transform(), dtype=dm_dataset.dtype)  # type: ignore[attr-defined]
            dataset.dm_subset = dm_dataset
            return dataset

        if dataset.task_type == OTXTaskType.DETECTION:
            return OTXTileDetTestDataset(dataset, tile_config, subset)
        if dataset.task_type in [OTXTaskType.ROTATED_DETECTION, OTXTaskType.INSTANCE_SEGMENTATION]:
            return OTXTileInstSegTestDataset(dataset, tile_config, subset)
        if dataset.task_type == OTXTaskType.SEMANTIC_SEGMENTATION:
            return OTXTileSemanticSegTestDataset(dataset, tile_config, subset)

        msg = f"Unsupported task type: {dataset.task_type} for tiling"
        raise NotImplementedError(msg)


class OTXTileDataset(OTXDataset):
    """OTX tile dataset base class.

    Args:
        dataset (OTXDataset): OTX dataset.
        tile_config (TilerConfig): Tile configuration.
    """

    def __init__(self, dataset: OTXDataset, tile_config: TileConfig, subset: Subset) -> None:
        super().__init__(
            dataset.dm_subset,
            dataset.transforms,
            dataset.max_refetch,
        )
        self.tile_config = tile_config
        self._dataset = dataset
        self._subset = subset
        self._tiling_config = TilingConfig(
            tile_height=tile_config.tile_size[0],
            tile_width=tile_config.tile_size[1],
            overlap_x=tile_config.overlap,
            overlap_y=tile_config.overlap,
        )
        self._tiling_transform = create_tiling_transform(self._tiling_config, threshold_drop_ann=0.5)
        self._filtering_transform = create_filtering_transform()

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

    @property
    def collate_fn(self) -> Callable:
        """Collate function from the original dataset."""
        return self._dataset.collate_fn

    def _get_item_impl(self, index: int) -> OTXSample | None:
        """Get item implementation from the original dataset."""
        return self._dataset._get_item_impl(index)

    def get_tiles(
        self,
        parent_idx: int,
    ) -> list[OTXSample]:
        """Retrieves tiles from the given image and dataset item.

        Args:
            image (np.ndarray): The input image.
            item (DatasetItem): The dataset item.
            parent_idx (int): The parent index. This is to keep track of the original dataset item index for merging.

        Returns:
            A tuple containing two lists:
            - tile_entities (list[OTXSample]): List of tile entities.
        """
        parent_slice_ds = self.dm_subset.slice(parent_idx, 1)  # type: ignore[attr-defined]
        tile_ds = parent_slice_ds.transform(  # type: ignore[attr-defined]
            self._tiling_transform,
            dtype=parent_slice_ds.dtype,  # type: ignore[attr-defined]
        )

        if self._subset == Subset.VALIDATION:
            # NOTE: filter validation tiles with annotations only to avoid evaluation on empty tiles.
            tile_ds = tile_ds.transform(self._filtering_transform, dtype=parent_slice_ds.dtype)  # type: ignore[attr-defined]

            # if tile dataset is empty it means objects are too big to fit in any tile, in this case include full image
            if len(tile_ds) == 0:
                tile_ds = parent_slice_ds

        tile_entities: list[OTXSample] = []
        for tile in tile_ds:
            # Fix datumaro HWC→CHW format issue for tile images
            tile.image = _ensure_chw_format(tile.image)
            # apply the same transforms as the original dataset
            object.__setattr__(tile.tile, "source_sample_idx", parent_idx)
            transformed_tile = self._apply_transforms(tile)
            if transformed_tile is None:
                msg = "Transformed tile is None"
                raise RuntimeError(msg)
            tile_entities.append(transformed_tile)
        return tile_entities


class OTXTileDetTestDataset(OTXTileDataset):
    """OTX tile detection test dataset.

    OTXTileDetTestDataset wraps a list of tiles (DetDataEntity) into a single TileDetDataEntity for testing/predicting.

    Args:
        dataset (OTXDetDataset): OTX detection dataset.
        tile_config (TilerConfig): Tile configuration.
    """

    def __init__(self, dataset: OTXDetectionDataset, tile_config: TileConfig, subset: Subset) -> None:
        super().__init__(dataset, tile_config, subset)

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
            the return of OTXSample. Nevertheless, in instances involving tiling, it becomes
            imperative to encapsulate tiles within a unified entity, namely TileDetDataEntity.
        """
        item = self.dm_subset[index]
        tile_entities = self.get_tiles(index)

        return TileDetDataEntity(
            num_tiles=len(tile_entities),
            entity_list=tile_entities,
            ori_img_info=item.img_info,  # type: ignore[attr-defined]
            ori_bboxes=item.bboxes,  # type: ignore[attr-defined]
            ori_labels=item.label,  # type: ignore[attr-defined]
        )


class OTXTileInstSegTestDataset(OTXTileDataset):
    """OTX tile inst-seg test dataset.

    OTXTileDetTestDataset wraps a list of tiles (TorchDataItem) into a single TileDetDataEntity
    for testing/predicting.

    Args:
        dataset (OTXInstanceSegDataset): OTX inst-seg dataset.
        tile_config (TilerConfig): Tile configuration.
    """

    def __init__(self, dataset: OTXInstanceSegDataset, tile_config: TileConfig, subset: Subset) -> None:
        super().__init__(dataset, tile_config, subset)

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
            the return of OTXSample. Nevertheless, in instances involving tiling, it becomes
            imperative to encapsulate tiles within a unified entity, namely TileInstSegDataEntity.
        """
        item = self.dm_subset[index]
        tile_entities = self.get_tiles(index)

        return TileInstSegDataEntity(
            num_tiles=len(tile_entities),
            entity_list=tile_entities,
            ori_img_info=item.img_info,  # type: ignore[attr-defined]
            ori_bboxes=item.bboxes,  # type: ignore[attr-defined]
            ori_labels=item.label,  # type: ignore[attr-defined]
            ori_masks=item.masks,  # type: ignore[attr-defined]
        )


class OTXTileSemanticSegTestDataset(OTXTileDataset):
    """OTX tile semantic-seg test dataset.

    OTXTileSemanticSegTestDataset wraps a list of tiles (SegDataEntity) into a single TileSegDataEntity
    for testing/predicting.

    Args:
        dataset (OTXSegmentationDataset): OTX semantic-seg dataset.
        tile_config (TilerConfig): Tile configuration.
    """

    def __init__(self, dataset: OTXSegmentationDataset, tile_config: TileConfig, subset: Subset) -> None:
        super().__init__(dataset, tile_config, subset)

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
        tile_entities = self.get_tiles(index)

        return TileSegDataEntity(
            num_tiles=len(tile_entities),
            entity_list=tile_entities,
            ori_img_info=item.img_info,  # type: ignore[attr-defined]
            ori_masks=item.masks,  # type: ignore[attr-defined]
        )
