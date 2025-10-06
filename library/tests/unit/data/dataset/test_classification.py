# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests of classification datasets."""

from unittest.mock import MagicMock

from otx.data.dataset.classification import (
    HLabelInfo,
    OTXHlabelClsDataset,
    OTXMulticlassClsDataset,
    OTXMultilabelClsDataset,
)
from otx.data.entity.torch import OTXDataItem


class TestOTXMulticlassClsDataset:
    def test_get_item(
        self,
        fxt_mock_dm_subset,
    ) -> None:
        dataset = OTXMulticlassClsDataset(
            dm_subset=fxt_mock_dm_subset,
            transforms=[lambda x: x],
            max_refetch=3,
        )
        assert isinstance(dataset[0], OTXDataItem)

    def test_get_item_from_bbox_dataset(
        self,
        fxt_mock_det_dm_subset,
    ) -> None:
        dataset = OTXMulticlassClsDataset(
            dm_subset=fxt_mock_det_dm_subset,
            transforms=[lambda x: x],
            max_refetch=3,
        )
        assert isinstance(dataset[0], OTXDataItem)


class TestOTXMultilabelClsDataset:
    def test_get_item(
        self,
        fxt_mock_dm_subset,
    ) -> None:
        dataset = OTXMultilabelClsDataset(
            dm_subset=fxt_mock_dm_subset,
            transforms=[lambda x: x],
            max_refetch=3,
        )
        assert isinstance(dataset[0], OTXDataItem)

    def test_get_item_from_bbox_dataset(
        self,
        fxt_mock_det_dm_subset,
    ) -> None:
        dataset = OTXMultilabelClsDataset(
            dm_subset=fxt_mock_det_dm_subset,
            transforms=[lambda x: x],
            max_refetch=3,
        )
        assert isinstance(dataset[0], OTXDataItem)


class TestOTXHlabelClsDataset:
    def test_add_ancestors(self, fxt_hlabel_dataset_subset):
        original_anns = fxt_hlabel_dataset_subset.get(id=0, subset="train").annotations
        assert len(original_anns) == 1

        hlabel_dataset = OTXHlabelClsDataset(
            dm_subset=fxt_hlabel_dataset_subset,
            transforms=MagicMock(),
        )
        # Added the ancestor
        adjusted_anns = hlabel_dataset.dm_subset.get(id=0, subset="train").annotations
        assert len(adjusted_anns) == 2

    def test_get_item(
        self,
        mocker,
        fxt_mock_dm_subset,
        fxt_mock_hlabelinfo,
    ) -> None:
        mocker.patch.object(HLabelInfo, "from_dm_label_groups", return_value=fxt_mock_hlabelinfo)
        dataset = OTXHlabelClsDataset(
            dm_subset=fxt_mock_dm_subset,
            transforms=[lambda x: x],
            max_refetch=3,
        )
        assert isinstance(dataset[0], OTXDataItem)

    def test_get_item_from_bbox_dataset(
        self,
        mocker,
        fxt_mock_det_dm_subset,
        fxt_mock_hlabelinfo,
    ) -> None:
        mocker.patch.object(HLabelInfo, "from_dm_label_groups", return_value=fxt_mock_hlabelinfo)
        dataset = OTXHlabelClsDataset(
            dm_subset=fxt_mock_det_dm_subset,
            transforms=[lambda x: x],
            max_refetch=3,
        )
        assert isinstance(dataset[0], OTXDataItem)
