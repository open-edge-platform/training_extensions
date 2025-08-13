# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from datumaro import Image
from datumaro.plugins.tiling.util import xywh_to_x1y1x2y2
from model_api.models import Model
from model_api.tilers import Tiler

from otx.config.data import TileConfig
from otx.data.dataset.base import OTXDataset
from otx.data.dataset.tile import OTXTileTrainDataset, OTXTileTransform


def test_tile_transform_consistency(mocker):
    # Test that OV tiler and PyTorch tile transform are consistent
    rng = np.random.default_rng()
    rnd_tile_size = rng.integers(low=100, high=500)
    rnd_tile_overlap = min(rng.random(), 0.9)
    image_size = rng.integers(low=1000, high=5000)
    np_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    dm_image = Image.from_numpy(np_image)

    mock_model = MagicMock(spec=Model)
    mocker.patch("model_api.tilers.tiler.Tiler.__init__", return_value=None)
    mocker.patch.multiple(Tiler, __abstractmethods__=set())

    tiler = Tiler(model=mock_model)
    tiler.tile_with_full_img = True
    tiler.tile_size = rnd_tile_size
    tiler.tiles_overlap = rnd_tile_overlap

    mocker.patch("otx.data.dataset.tile.OTXTileTransform.__init__", return_value=None)
    tile_transform = OTXTileTransform()
    tile_transform._tile_size = (rnd_tile_size, rnd_tile_size)
    tile_transform._overlap = (rnd_tile_overlap, rnd_tile_overlap)
    tile_transform.with_full_img = True

    dm_rois = [xywh_to_x1y1x2y2(*roi) for roi in tile_transform._extract_rois(dm_image)]
    ov_tiler_rois = tiler._tile(np_image)

    assert len(dm_rois) == len(ov_tiler_rois)
    for dm_roi in dm_rois:
        assert list(dm_roi) in ov_tiler_rois


def test_empty_tiled_dataset_raises_value_error():
    """Test that OTXTileTrainDataset raises ValueError when tiled dataset is empty."""
    # Create a mock dataset with empty dm_subset after tiling
    mock_dataset = MagicMock(spec=OTXDataset)
    mock_dm_subset = MagicMock()

    # Mock the transform method to return an empty dataset
    empty_dm_subset = MagicMock()
    empty_dm_subset.__len__.return_value = 0
    empty_dm_subset.filter.return_value = empty_dm_subset

    mock_dm_subset.transform.return_value = empty_dm_subset
    mock_dataset.dm_subset = mock_dm_subset

    # Create tile config with small tile size to trigger the empty dataset scenario
    tile_config = TileConfig(
        enable_tiler=True,
        tile_size=(10, 10),  # Very small tile size
        overlap=0.1,
        with_full_img=False,
    )

    # Test that ValueError is raised with the expected message
    expected_msg = (
        "Tiled dataset is empty. This is likely because the tile_size \\(\\(10, 10\\)\\) "
        "is too small, causing all annotations to be discarded. "
        "\\*\\*Try increasing the tile_size\\.\\*\\*"
    )
    with pytest.raises(ValueError, match=expected_msg):
        OTXTileTrainDataset(dataset=mock_dataset, tile_config=tile_config)


def test_empty_tiled_dataset_raises_value_error_different_tile_size():
    """Test that OTXTileTrainDataset raises ValueError with different tile size in message."""
    # Create a mock dataset with empty dm_subset after tiling
    mock_dataset = MagicMock(spec=OTXDataset)
    mock_dm_subset = MagicMock()

    # Mock the transform method to return an empty dataset
    empty_dm_subset = MagicMock()
    empty_dm_subset.__len__.return_value = 0
    empty_dm_subset.filter.return_value = empty_dm_subset

    mock_dm_subset.transform.return_value = empty_dm_subset
    mock_dataset.dm_subset = mock_dm_subset

    # Create tile config with different tile size
    tile_config = TileConfig(
        enable_tiler=True,
        tile_size=(50, 50),  # Different tile size
        overlap=0.2,
        with_full_img=True,
    )

    # Test that ValueError is raised with the expected message containing the tile size
    expected_msg = (
        "Tiled dataset is empty. This is likely because the tile_size \\(\\(50, 50\\)\\) "
        "is too small, causing all annotations to be discarded. "
        "\\*\\*Try increasing the tile_size\\.\\*\\*"
    )
    with pytest.raises(ValueError, match=expected_msg):
        OTXTileTrainDataset(dataset=mock_dataset, tile_config=tile_config)
