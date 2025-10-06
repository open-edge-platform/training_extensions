# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module defines OTXDatasets."""

from .anomaly import OTXAnomalyDataset
from .classification import OTXHlabelClsDataset, OTXMulticlassClsDataset, OTXMultilabelClsDataset
from .detection import OTXDetectionDataset
from .instance_segmentation import OTXInstanceSegDataset
from .keypoint_detection import OTXKeypointDetectionDataset
from .segmentation import OTXSegmentationDataset
from .tile import OTXTileDatasetFactory

__all__ = [
    "OTXAnomalyDataset",
    "OTXMulticlassClsDataset",
    "OTXHlabelClsDataset",
    "OTXMultilabelClsDataset",
    "OTXDetectionDataset",
    "OTXInstanceSegDataset",
    "OTXKeypointDetectionDataset",
    "OTXSegmentationDataset",
    "OTXTileDatasetFactory",
]
