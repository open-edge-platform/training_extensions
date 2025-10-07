# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from importlib_resources import files
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from otx.config.data import (
    SubsetConfig,
    TileConfig,
)
from otx.data import module as target_file
from otx.data.module import (
    DeviceType,
    OTXDataModule,
    OTXTaskType,
)

if TYPE_CHECKING:
    from datumaro.components.dataset import Dataset as DmDataset


class TestModule:
    @pytest.fixture()
    def fxt_config(self) -> DictConfig:
        train_subset = MagicMock(spec=SubsetConfig)
        train_subset.sampler = DictConfig(
            {"class_path": "torch.utils.data.RandomSampler", "init_args": {"num_samples": 4}},
        )
        train_subset.num_workers = 0
        train_subset.batch_size = 4
        train_subset.input_size = None
        train_subset.subset_name = "train_1"
        train_subset.transforms = []
        val_subset = MagicMock(spec=SubsetConfig)
        val_subset.sampler = DictConfig(
            {"class_path": "torch.utils.data.RandomSampler", "init_args": {"num_samples": 3}},
        )
        val_subset.num_workers = 0
        val_subset.batch_size = 3
        val_subset.input_size = None
        val_subset.subset_name = "val_1"
        test_subset = MagicMock(spec=SubsetConfig)
        test_subset.sampler = DictConfig(
            {"class_path": "torch.utils.data.RandomSampler", "init_args": {"num_samples": 3}},
        )
        test_subset.num_workers = 0
        test_subset.batch_size = 1
        test_subset.input_size = None
        test_subset.subset_name = "test_1"
        tile_config = MagicMock(spec=TileConfig)
        tile_config.enable_tiler = False

        mock = MagicMock(spec=DictConfig)
        mock.task = "MULTI_LABEL_CLS"
        mock.data_format = "coco_instances"
        mock.data_root = "."
        mock.train_subset = train_subset
        mock.val_subset = val_subset
        mock.test_subset = test_subset
        mock.tile_config = tile_config

        return mock

    @pytest.fixture()
    def mock_dm_dataset(self, mocker) -> MagicMock:
        return mocker.patch("otx.data.module.DmDataset.import_from")

    @pytest.fixture()
    def mock_otx_dataset_factory(self, mocker) -> MagicMock:
        return mocker.patch("otx.data.module.OTXDatasetFactory")

    @pytest.fixture()
    def mock_data_filtering(self, mocker) -> MagicMock:
        def func(
            dataset: DmDataset,
            data_format: str,
            unannotated_items_ratio: float,
            task: OTXTaskType,
            ignore_index: int | None,
        ) -> DmDataset:
            del data_format
            del unannotated_items_ratio
            del ignore_index
            del task
            return dataset

        return mocker.patch("otx.data.module.pre_filtering", side_effect=func)

    @pytest.mark.parametrize(
        "task",
        [
            OTXTaskType.MULTI_CLASS_CLS,
            OTXTaskType.MULTI_LABEL_CLS,
            OTXTaskType.H_LABEL_CLS,
            OTXTaskType.DETECTION,
            OTXTaskType.SEMANTIC_SEGMENTATION,
            OTXTaskType.INSTANCE_SEGMENTATION,
        ],
    )
    def test_init(
        self,
        mock_dm_dataset,
        mock_otx_dataset_factory,
        mock_data_filtering,
        task,
        fxt_config,
    ) -> None:
        # Dataset will have "train_0", "train_1", "val_0", ..., "test_1" subsets
        mock_dm_subsets = {f"{name}_{idx}": MagicMock() for name in ["train", "val", "test"] for idx in range(2)}
        mock_dm_dataset.return_value.subsets.return_value = mock_dm_subsets

        module = OTXDataModule(
            task=task,
            data_format=fxt_config.data_format,
            data_root=fxt_config.data_root,
            train_subset=fxt_config.train_subset,
            val_subset=fxt_config.val_subset,
            test_subset=fxt_config.test_subset,
            input_size=(240, 240),
        )

        assert module.train_dataloader().batch_size == 4
        assert module.val_dataloader().batch_size == 3
        assert module.test_dataloader().batch_size == 1
        assert mock_otx_dataset_factory.create.call_count == 3
        assert fxt_config.train_subset.input_size == (240, 240)
        assert fxt_config.val_subset.input_size == (240, 240)
        assert fxt_config.test_subset.input_size == (240, 240)

    def test_init_input_size(
        self,
        mock_dm_dataset,
        mock_otx_dataset_factory,
        mock_data_filtering,
        fxt_config,
    ) -> None:
        # Dataset will have "train_0", "train_1", "val_0", ..., "test_1" subsets
        mock_dm_subsets = {f"{name}_{idx}": MagicMock() for name in ["train", "val", "test"] for idx in range(2)}
        mock_dm_dataset.return_value.subsets.return_value = mock_dm_subsets
        fxt_config.train_subset.input_size = None
        fxt_config.val_subset.input_size = None
        fxt_config.test_subset.input_size = (800, 800)

        OTXDataModule(
            task=OTXTaskType.MULTI_CLASS_CLS,
            data_format=fxt_config.data_format,
            data_root=fxt_config.data_root,
            train_subset=fxt_config.train_subset,
            val_subset=fxt_config.val_subset,
            test_subset=fxt_config.test_subset,
            input_size=(1200, 1200),
        )

        assert fxt_config.train_subset.input_size == (1200, 1200)
        assert fxt_config.val_subset.input_size == (1200, 1200)
        assert fxt_config.test_subset.input_size == (800, 800)

    @pytest.fixture()
    def mock_adapt_input_size_to_dataset(self, mocker) -> MagicMock:
        return mocker.patch.object(target_file, "adapt_input_size_to_dataset", return_value=(1234, 1234))

    @pytest.fixture()
    def fxt_real_tv_cls_config(self) -> DictConfig:
        cfg_path = files("otx") / "recipe" / "_base_" / "data" / "torchvision_base.yaml"
        cfg = OmegaConf.load(cfg_path)
        cfg.data_root = "."
        cfg.train_subset.subset_name = "train"
        cfg.train_subset.num_workers = 0
        cfg.train_subset.input_size = None
        cfg.val_subset.subset_name = "val"
        cfg.val_subset.num_workers = 0
        cfg.val_subset.input_size = None
        cfg.test_subset.subset_name = "test"
        cfg.test_subset.num_workers = 0
        cfg.test_subset.input_size = None
        cfg.tile_config = {}
        cfg.tile_config.enable_tiler = False
        cfg.auto_num_workers = False
        cfg.device = "auto"
        return cfg

    def test_hparams_initial_is_loggable(
        self,
        mock_dm_dataset,
        mock_otx_dataset_factory,
        mock_data_filtering,
        fxt_real_tv_cls_config,
        tmpdir,
    ) -> None:
        # Dataset will have "train", "val", and "test" subsets
        mock_dm_subsets = {name: MagicMock() for name in ["train", "val", "test"]}
        mock_dm_dataset.return_value.subsets.return_value = mock_dm_subsets
        module = OTXDataModule(**fxt_real_tv_cls_config, input_size=(240, 240))
        logger = CSVLogger(tmpdir)
        logger.log_hyperparams(module.hparams_initial)
        logger.save()

        hparams_path = Path(logger.log_dir) / "hparams.yaml"
        assert hparams_path.exists()

    def test_from_otx_datasets_basic(self, mocker) -> None:
        """Test from_otx_datasets with minimal configuration."""
        # Create mock datasets
        mock_train = MagicMock()
        mock_train.label_info = MagicMock()
        mock_train.task = OTXTaskType.MULTI_CLASS_CLS
        mock_train.data_format = "coco"
        mock_train.image_color_channel = "RGB"
        mock_train.transforms = []
        mock_train.__iter__ = lambda self: iter([MagicMock(img_info={"img_shape": (224, 224)})])

        mock_val = MagicMock()
        mock_val.label_info = mock_train.label_info
        mock_val.transforms = []

        mock_test = MagicMock()
        mock_test.label_info = mock_train.label_info
        mock_test.transforms = []

        # Mock _get_default_subset_config to return a proper SubsetConfig
        mock_subset_config = MagicMock(spec=SubsetConfig)
        mock_subset_config.batch_size = 32
        mock_subset_config.num_workers = 2
        mock_subset_config.sampler = DictConfig({"class_path": "torch.utils.data.RandomSampler"})
        mock_subset_config.transforms = []

        mocker.patch.object(
            OTXDataModule,
            "_get_default_subset_config",
            return_value=mock_subset_config,
        )

        # Create module from datasets
        module = OTXDataModule.from_otx_datasets(
            train_dataset=mock_train,
            val_dataset=mock_val,
            test_dataset=mock_test,
        )

        # Assertions
        assert module.subsets["train"] == mock_train
        assert module.subsets["val"] == mock_val
        assert module.subsets["test"] == mock_test
        assert module.label_info == mock_train.label_info
        assert module.task == OTXTaskType.MULTI_CLASS_CLS
        assert module.input_size == (224, 224)

    def test_from_otx_datasets_with_custom_configs(self, mocker) -> None:
        """Test from_otx_datasets with custom subset configurations."""
        # Create mock datasets
        mock_train = MagicMock()
        mock_train.label_info = MagicMock()
        mock_train.task = OTXTaskType.DETECTION
        mock_train.data_format = "coco"
        mock_train.image_color_channel = "RGB"
        mock_train.transforms = []
        mock_train.__iter__ = lambda self: iter([MagicMock(img_info={"img_shape": (640, 640)})])

        mock_val = MagicMock()
        mock_val.label_info = mock_train.label_info
        mock_val.transforms = []

        # Create custom configs
        train_config = MagicMock(spec=SubsetConfig)
        train_config.batch_size = 16
        train_config.num_workers = 4
        train_config.sampler = DictConfig({"class_path": "torch.utils.data.RandomSampler"})
        train_config.transforms = []

        val_config = MagicMock(spec=SubsetConfig)
        val_config.batch_size = 8
        val_config.num_workers = 2
        val_config.sampler = DictConfig({"class_path": "torch.utils.data.RandomSampler"})
        val_config.transforms = []

        # Mock default config for test subset
        mock_test_config = MagicMock(spec=SubsetConfig)
        mock_test_config.batch_size = 1
        mock_test_config.num_workers = 0
        mock_test_config.sampler = DictConfig({"class_path": "torch.utils.data.RandomSampler"})
        mock_test_config.transforms = []

        mocker.patch.object(
            OTXDataModule,
            "_get_default_subset_config",
            return_value=mock_test_config,
        )

        # Create module with custom configs
        module = OTXDataModule.from_otx_datasets(
            train_dataset=mock_train,
            val_dataset=mock_val,
            train_subset=train_config,
            val_subset=val_config,
        )

        # Assertions
        assert module.train_subset == train_config
        assert module.val_subset == val_config
        assert module.test_subset == mock_test_config
        assert module.task == OTXTaskType.DETECTION

    def test_from_otx_datasets_without_test(self, mocker) -> None:
        """Test from_otx_datasets when test_dataset is None (uses val as test)."""
        # Create mock datasets
        mock_train = MagicMock()
        mock_train.label_info = MagicMock()
        mock_train.task = OTXTaskType.SEMANTIC_SEGMENTATION
        mock_train.data_format = "coco"
        mock_train.image_color_channel = "RGB"
        mock_train.transforms = []
        mock_train.__iter__ = lambda self: iter([MagicMock(img_info={"img_shape": (512, 512)})])

        mock_val = MagicMock()
        mock_val.label_info = mock_train.label_info
        mock_val.transforms = []

        # Mock _get_default_subset_config
        mock_subset_config = MagicMock(spec=SubsetConfig)
        mock_subset_config.batch_size = 8
        mock_subset_config.num_workers = 2
        mock_subset_config.sampler = DictConfig({"class_path": "torch.utils.data.RandomSampler"})
        mock_subset_config.transforms = []

        mocker.patch.object(
            OTXDataModule,
            "_get_default_subset_config",
            return_value=mock_subset_config,
        )

        # Create module without test dataset
        module = OTXDataModule.from_otx_datasets(
            train_dataset=mock_train,
            val_dataset=mock_val,
            test_dataset=None,  # Explicitly None
        )

        # Assertions - test should use val dataset
        assert module.subsets["train"] == mock_train
        assert module.subsets["val"] == mock_val
        assert module.subsets["test"] == mock_val  # Should be val dataset
        assert module.task == OTXTaskType.SEMANTIC_SEGMENTATION

    def test_from_otx_datasets_label_info_mismatch(self, mocker) -> None:
        """Test from_otx_datasets raises error when label_info doesn't match."""
        # Create mock datasets with mismatched label_info
        mock_train = MagicMock()
        mock_train.label_info = MagicMock()
        mock_train.task = OTXTaskType.MULTI_CLASS_CLS
        mock_train.data_format = "coco"
        mock_train.image_color_channel = "RGB"
        mock_train.transforms = []
        mock_train.__iter__ = lambda self: iter([MagicMock(img_info={"img_shape": (224, 224)})])

        mock_val = MagicMock()
        mock_val.label_info = MagicMock()  # Different label_info
        mock_val.transforms = []

        # Should raise ValueError due to label_info mismatch
        with pytest.raises(ValueError, match="All data meta infos of provided datasets should be the same"):
            OTXDataModule.from_otx_datasets(
                train_dataset=mock_train,
                val_dataset=mock_val,
            )

    def test_from_otx_datasets_with_normalization(self, mocker) -> None:
        """Test from_otx_datasets correctly extracts normalization parameters."""
        from torchvision.transforms.v2 import Normalize

        # Create mock dataset with Normalize transform
        normalize_transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mock_train = MagicMock()
        mock_train.label_info = MagicMock()
        mock_train.task = OTXTaskType.MULTI_CLASS_CLS
        mock_train.data_format = "coco"
        mock_train.image_color_channel = "RGB"
        mock_train.transforms = MagicMock()
        mock_train.transforms.transforms = [normalize_transform]
        mock_train.__iter__ = lambda self: iter([MagicMock(img_info={"img_shape": (224, 224)})])

        mock_val = MagicMock()
        mock_val.label_info = mock_train.label_info
        mock_val.transforms = []

        # Mock _get_default_subset_config
        mock_subset_config = MagicMock(spec=SubsetConfig)
        mock_subset_config.batch_size = 32
        mock_subset_config.num_workers = 2
        mock_subset_config.sampler = DictConfig({"class_path": "torch.utils.data.RandomSampler"})
        mock_subset_config.transforms = []

        mocker.patch.object(
            OTXDataModule,
            "_get_default_subset_config",
            return_value=mock_subset_config,
        )

        # Create module
        module = OTXDataModule.from_otx_datasets(
            train_dataset=mock_train,
            val_dataset=mock_val,
        )

        # Assertions - normalization params should be extracted
        assert module.input_mean == normalize_transform.mean
        assert module.input_std == normalize_transform.std

    def test_from_otx_datasets_with_auto_num_workers(self, mocker) -> None:
        """Test from_otx_datasets with auto_num_workers enabled."""
        # Create mock datasets
        mock_train = MagicMock()
        mock_train.label_info = MagicMock()
        mock_train.task = OTXTaskType.DETECTION
        mock_train.data_format = "coco"
        mock_train.image_color_channel = "RGB"
        mock_train.transforms = []
        mock_train.__iter__ = lambda self: iter([MagicMock(img_info={"img_shape": (640, 640)})])

        mock_val = MagicMock()
        mock_val.label_info = mock_train.label_info
        mock_val.transforms = []

        # Mock _get_default_subset_config
        mock_subset_config = MagicMock(spec=SubsetConfig)
        mock_subset_config.batch_size = 32
        mock_subset_config.num_workers = 2
        mock_subset_config.sampler = DictConfig({"class_path": "torch.utils.data.RandomSampler"})
        mock_subset_config.transforms = []

        mocker.patch.object(
            OTXDataModule,
            "_get_default_subset_config",
            return_value=mock_subset_config,
        )

        # Create module with auto_num_workers
        module = OTXDataModule.from_otx_datasets(
            train_dataset=mock_train,
            val_dataset=mock_val,
            auto_num_workers=True,
        )

        # Assertions
        assert module.auto_num_workers is True
        assert module.device == DeviceType.auto  # Default value
