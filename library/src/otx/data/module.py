# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LightningDataModule extension for OTX."""

from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING

from datumaro import Dataset as DmDataset
from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, RandomSampler
from torchvision.transforms.v2 import Normalize

from otx.config.data import TileConfig
from otx.data.dataset.tile import OTXTileDatasetFactory
from otx.data.factory import OTXDatasetFactory
from otx.data.utils import adapt_input_size_to_dataset, adapt_tile_config, get_adaptive_num_workers, instantiate_sampler
from otx.data.utils.pre_filtering import pre_filtering
from otx.types.device import DeviceType
from otx.types.image import ImageColorChannel
from otx.types.label import LabelInfo
from otx.types.task import OTXTaskType

if TYPE_CHECKING:
    from lightning.pytorch.utilities.parsing import AttributeDict

    from otx.config.data import SubsetConfig
    from otx.data.dataset.base import OTXDataset


class OTXDataModule(LightningDataModule):
    """LightningDataModule extension for OTX.

    Handles data loading, transformation, and preparation for OTX pipelines.

    Args:
        task (OTXTaskType): Task type (e.g., classification, detection).
        data_root (str): Root directory of the dataset.
        data_format (str, optional): Data format (e.g., 'coco', 'voc'). Defaults to None.
        train_subset (SubsetConfig, optional): Training subset configuration. Defaults to None.
        val_subset (SubsetConfig, optional): Validation subset configuration. Defaults to None.
        test_subset (SubsetConfig, optional): Test subset configuration. Defaults to None.
        tile_config (TileConfig, optional): Tiling configuration. Defaults to TileConfig(enable_tiler=False).
        image_color_channel (ImageColorChannel, optional): Image color channel. Defaults to ImageColorChannel.RGB.
        include_polygons (bool, optional): Whether to include polygons. Defaults to False.
        ignore_index (int, optional): Ignore index for segmentation. Defaults to 255.
        unannotated_items_ratio (float, optional): Ratio of unannotated items. Defaults to 0.0.
        auto_num_workers (bool, optional): Automatically determine number of workers. Defaults to False.
        device (DeviceType, optional): Device type ('cpu', 'gpu', etc.). Defaults to DeviceType.auto.
        input_size (tuple[int, int] | None, optional): Final image/video shape after transformation. Defaults to None.

    Note:
        To create an OTXDataModule from pre-constructed datasets, use the `from_otx_datasets` class method.
    """

    def __init__(
        self,
        task: OTXTaskType,
        data_root: str,
        data_format: str | None = None,
        train_subset: SubsetConfig | None = None,
        val_subset: SubsetConfig | None = None,
        test_subset: SubsetConfig | None = None,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        include_polygons: bool = False,
        ignore_index: int = 255,
        unannotated_items_ratio: float = 0.0,
        auto_num_workers: bool = False,
        device: DeviceType = DeviceType.auto,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        """Constructor."""
        super().__init__()

        self.task = task
        self.data_format = data_format
        self.data_root = data_root

        self.train_subset = train_subset if train_subset is not None else self._get_default_subset_config("train")
        self.val_subset = val_subset if val_subset is not None else self._get_default_subset_config("val")
        self.test_subset = test_subset if test_subset is not None else self._get_default_subset_config("test")

        self.tile_config = tile_config

        self.image_color_channel = image_color_channel
        self.include_polygons = include_polygons
        self.ignore_index = ignore_index
        self.unannotated_items_ratio = unannotated_items_ratio

        self.auto_num_workers = auto_num_workers
        self.device = device

        self.subsets: dict[str, OTXDataset] = {}
        self.save_hyperparameters(ignore=["input_size"])

        dataset = DmDataset.import_from(self.data_root, format=self.data_format)
        if self.task != OTXTaskType.H_LABEL_CLS and not (
            self.task == OTXTaskType.KEYPOINT_DETECTION and self.data_format == "arrow"
        ):
            dataset = pre_filtering(
                dataset,
                self.data_format,
                self.unannotated_items_ratio,
                self.task,
                ignore_index=self.ignore_index if self.task == "SEMANTIC_SEGMENTATION" else None,
            )

        if input_size is not None and not isinstance(input_size, (tuple, list)):
            msg = f"input_size should be tuple/list of ints or 'auto', but got {input_size}"
            raise ValueError(msg)

        elif input_size is not None:
            # override input_size to all subset configs when it is given
            for subset_cfg in [self.train_subset, self.val_subset, self.test_subset]:
                subset_cfg.input_size = input_size  # type: ignore[assignment]

        # Extract mean and std from Normalize transform
        self.input_mean, self.input_std = self._extract_normalization_params(self.train_subset.transforms)
        self.input_size = input_size

        if self.tile_config.enable_tiler and self.tile_config.enable_adaptive_tiling:
            adapt_tile_config(self.tile_config, dataset=dataset, task=self.task)

        self._setup_otx_dataset(dataset)

    def _setup_otx_dataset(self, dataset: DmDataset) -> None:
        """Setup OTXDataset from Datumaro Dataset object.

        Args:
            dataset: Datumaro Dataset object.
        Returns: None
        """
        config_mapping = {
            self.train_subset.subset_name: self.train_subset,
            self.val_subset.subset_name: self.val_subset,
            self.test_subset.subset_name: self.test_subset,
        }

        if self.auto_num_workers:
            if self.device not in [DeviceType.gpu, DeviceType.auto]:
                log.warning(
                    "Only GPU device type support auto_num_workers. "
                    f"Current deveice type is {self.device!s}. auto_num_workers is skipped.",
                )
            elif (num_workers := get_adaptive_num_workers()) is not None:
                for subset_name, subset_config in config_mapping.items():
                    log.info(
                        f"num_workers of {subset_name} subset is changed : "
                        f"{subset_config.num_workers} -> {num_workers}",
                    )
                    subset_config.num_workers = num_workers

        label_infos: list[LabelInfo] = []
        for name, dm_subset in dataset.subsets().items():
            if name not in config_mapping:
                log.warning(f"{name} is not available. Skip it")
                continue

            dataset = OTXDatasetFactory.create(
                task=self.task,
                dm_subset=dm_subset.as_dataset(),
                cfg_subset=config_mapping[name],
                data_format=self.data_format,
                image_color_channel=self.image_color_channel,
                include_polygons=self.include_polygons,
                ignore_index=self.ignore_index,
            )

            if self.tile_config.enable_tiler:
                dataset = OTXTileDatasetFactory.create(
                    task=self.task,
                    dataset=dataset,
                    tile_config=self.tile_config,
                )
            self.subsets[name] = dataset
            label_infos += [self.subsets[name].label_info]
            log.info(f"Add name: {name}, self.subsets: {self.subsets}")

        if self._is_meta_info_valid(label_infos) is False:
            msg = "All data meta infos of subsets should be the same."
            raise ValueError(msg)

        self.label_info = next(iter(label_infos))

    def _extract_normalization_params(self, transforms_source: list | None) -> tuple[tuple, tuple]:
        """Extract mean and std from transforms.

        Args:
            transforms_source: List of transforms or None.

        Returns:
            Tuple of (mean, std) tuples.
        """
        mean = (0.0, 0.0, 0.0)
        std = (1.0, 1.0, 1.0)

        if transforms_source is not None:
            for transform in transforms_source:
                if isinstance(transform, dict) and "Normalize" in transform.get("class_path", ""):
                    # CLI case with jsonargparse
                    mean = transform["init_args"].get("mean", (0.0, 0.0, 0.0))
                    std = transform["init_args"].get("std", (1.0, 1.0, 1.0))
                    break

                if isinstance(transform, Normalize):
                    # torchvision.transforms case
                    mean = transform.mean
                    std = transform.std
                    break

        return mean, std

    @classmethod
    def from_otx_datasets(
        cls,
        task: OTXTaskType,
        train_dataset: OTXDataset,
        val_dataset: OTXDataset,
        test_dataset: OTXDataset | None = None,
        train_subset: SubsetConfig | None = None,
        val_subset: SubsetConfig | None = None,
        test_subset: SubsetConfig | None = None,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        auto_num_workers: bool = False,
        device: DeviceType = DeviceType.auto,
    ) -> OTXDataModule:
        """Create an OTXDataModule from pre-constructed OTXDataset instances.

        This is a factory method that provides a clean way to create OTXDataModule instances
        when you already have constructed datasets, without needing to provide data_root,
        data_format, or other data loading parameters.

        Args:
            task (OTXTaskType): The type of task (e.g., classification, detection).
            datasets (dict[str, OTXDataset]): Pre-constructed OTX datasets.
                Keys should be subset names ('train', 'val', 'test').
                Values should be OTXDataset instances.
            train_subset (SubsetConfig | None, optional): Configuration for the training dataloader.
                If None, default configuration will be used. Defaults to None.
            val_subset (SubsetConfig | None, optional): Configuration for the validation dataloader.
                If None, default configuration will be used. Defaults to None.
            test_subset (SubsetConfig | None, optional): Configuration for the test dataloader.
                If None, default configuration will be used. Defaults to None.
            tile_config (TileConfig, optional): Configuration for tiling.
                Defaults to TileConfig(enable_tiler=False).
            image_color_channel (ImageColorChannel, optional): Color channel configuration for images.
                Defaults to ImageColorChannel.RGB.
            auto_num_workers (bool, optional): Whether to automatically determine the number of workers.
                Defaults to False.
            device (DeviceType, optional): Device type to use (e.g., 'cpu', 'gpu').
                Defaults to DeviceType.auto.
            input_size (tuple[int, int] | str, optional): Final image or video shape after transformation.
                Can be "auto" to determine size automatically. Defaults to "auto".

        Returns:
            OTXDataModule: Configured data module with the provided datasets.

        Raises:
            ValueError: If datasets dictionary is empty or datasets have inconsistent metadata.

        Examples:
            >>> from otx.data.module import OTXDataModule
            >>> from otx.types.task import OTXTaskType
            >>>
            >>> # Create datamodule with minimal configuration
            >>> datasets = {
            ...     "train": my_train_dataset,
            ...     "val": my_val_dataset,
            ...     "test": my_test_dataset,
            ... }
            >>> datamodule = OTXDataModule.from_otx_datasets(
            ...     task=OTXTaskType.DETECTION,
            ...     datasets=datasets,
            ...     input_size=(640, 640),
            ... )
            >>>
            >>> # Create datamodule with custom subset configurations
            >>> from otx.config.data import SubsetConfig
            >>> train_config = SubsetConfig(
            ...     batch_size=64,
            ...     subset_name="train",
            ...     transforms=[],
            ...     num_workers=8,
            ... )
            >>> datamodule = OTXDataModule.from_otx_datasets(
            ...     task=OTXTaskType.DETECTION,
            ...     datasets=datasets,
            ...     train_subset=train_config,
            ...     input_size=(640, 640),
            ... )
        """
        # Validate label info consistency across datasets
        if test_dataset is None:
            test_dataset = val_dataset

        if not all(label_info == train_dataset.label_info for label_info in [val_dataset.label_info,
                                                                              test_dataset.label_info]):
            msg = "All data meta infos of provided datasets should be the same."
            raise ValueError(msg)

        # Create instance
        instance = cls.__new__(cls)
        LightningDataModule.__init__(instance)

        # Set basic attributes
        instance.task = task
        instance.data_format = train_dataset.data_format
        instance.data_root = None
        instance.train_subset = train_subset if train_subset is not None else instance._get_default_subset_config("train")
        instance.val_subset = val_subset if val_subset is not None else instance._get_default_subset_config("val")
        instance.test_subset = test_subset if test_subset is not None else instance._get_default_subset_config("test")
        instance.tile_config = tile_config
        instance.image_color_channel = train_dataset.image_color_channel
        instance.include_polygons = False
        instance.ignore_index = 255
        instance.unannotated_items_ratio = 0.0
        instance.auto_num_workers = auto_num_workers
        instance.device = device

        # Store datasets and label info
        instance.subsets = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
        }
        instance.label_info = train_dataset.label_info

        # derive image_size from dataset
        # assume that data uses fixed size during transforms
        example_item = next(iter(train_dataset))
        input_size = example_item.img_info["img_shape"]
        instance.input_size = input_size

        # Extract normalization parameters from first dataset's transforms if available
        transforms_to_extract = None

        if hasattr(train_dataset, "transforms") and train_dataset.transforms is not None:
            transforms_list = train_dataset.transforms
            # Try to extract transforms list from Compose object
            if hasattr(transforms_list, "transforms") and isinstance(
                getattr(transforms_list, "transforms", None), list
            ):
                transforms_to_extract = transforms_list.transforms  # type: ignore[union-attr]
            elif isinstance(transforms_list, list):
                transforms_to_extract = transforms_list

        instance.input_mean, instance.input_std = instance._extract_normalization_params(transforms_to_extract)

        # override transforms in subset_config based on provided datasets
        instance.train_subset.transforms = train_dataset.transforms
        instance.val_subset.transforms = val_dataset.transforms
        instance.test_subset.transforms = test_dataset.transforms

        # Save hyperparameters
        instance.save_hyperparameters(
            ignore=["datasets", "input_size"],
            logger=False,
        )

        return instance

    def _get_default_subset_config(self, subset_name: str) -> SubsetConfig:
        """Create a default SubsetConfig for a given subset when not provided.

        This method loads the configuration from the base YAML files in
        otx/recipe/_base_/data based on the task type.

        Args:
            subset_name: Name of the subset ('train', 'val', or 'test').

        Returns:
            SubsetConfig: Default configuration for the subset loaded from base config.
        """
        from pathlib import Path
        from omegaconf import OmegaConf
        from otx.config.data import SubsetConfig

        # Check if the subset exists in our datasets
        if subset_name not in self.subsets:
            msg = f"Subset '{subset_name}' not found in provided datasets. Available: {list(self.subsets.keys())}"
            raise ValueError(msg)

        # Map task type to config file name
        task_to_data_config_file = {
            OTXTaskType.ANOMALY: "anomaly.yaml",
            OTXTaskType.ANOMALY_CLASSIFICATION: "anomaly.yaml",
            OTXTaskType.ANOMALY_DETECTION: "anomaly.yaml",
            OTXTaskType.ANOMALY_SEGMENTATION: "anomaly.yaml",
            OTXTaskType.MULTI_CLASS_CLS: "classification.yaml",
            OTXTaskType.MULTI_LABEL_CLS: "classification.yaml",
            OTXTaskType.H_LABEL_CLS: "classification.yaml",
            OTXTaskType.DETECTION: "detection.yaml",
            OTXTaskType.ROTATED_DETECTION: "detection.yaml",
            OTXTaskType.INSTANCE_SEGMENTATION: "instance_segmentation.yaml",
            OTXTaskType.SEMANTIC_SEGMENTATION: "semantic_segmentation.yaml",
            OTXTaskType.KEYPOINT_DETECTION: "keypoint_detection.yaml",
        }

        config_file = task_to_data_config_file.get(self.task)
        if config_file is None:
            msg = f"No base config file found for task type: {self.task}"
            raise ValueError(msg)

        # Load the YAML configuration
        base_path = Path(__file__).parent.parent / "recipe" / "_base_" / "data" / config_file
        if not base_path.exists():
            msg = f"Base config file not found: {base_path}"
            raise FileNotFoundError(msg)

        # Load and parse the YAML
        config_dict = OmegaConf.load(base_path)

        # Get the subset configuration
        subset_key = f"{subset_name}_subset"
        if subset_key not in config_dict:
            msg = f"Subset '{subset_key}' not found in config file {config_file}"
            raise ValueError(msg)

        # Extract subset config and convert to container (dict)
        subset_config_dict = OmegaConf.to_container(config_dict[subset_key], resolve=True)  # type: ignore[index]
        subset_config_dict["input_size"] = config_dict["input_size"]
        # Create structured config from dict
        config = OmegaConf.structured(SubsetConfig)
        config = OmegaConf.merge(config, subset_config_dict)

        # Convert to Python object
        subset_config: SubsetConfig = OmegaConf.to_object(config)  # type: ignore[assignment]

        return subset_config

    def _is_meta_info_valid(self, label_infos: list[LabelInfo]) -> bool:
        """Check whether there are mismatches in the metainfo for the all subsets."""
        return bool(all(label_info == label_infos[0] for label_info in label_infos))

    def _get_dataset(self, subset: str) -> OTXDataset:
        if (dataset := self.subsets.get(subset)) is None:
            msg = f"Dataset has no '{subset}'. Available subsets = {list(self.subsets.keys())}"
            raise KeyError(msg)
        return dataset

    def train_dataloader(self) -> DataLoader:
        """Get train dataloader."""
        if self.train_subset is None:
            # Use default configuration when datasets are provided directly
            self.train_subset = self._get_default_subset_config("train")

        config = self.train_subset
        dataset = self._get_dataset(config.subset_name)
        sampler = instantiate_sampler(config.sampler, dataset=dataset, batch_size=config.batch_size)

        common_args = {
            "dataset": dataset,
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "pin_memory": True,
            "collate_fn": dataset.collate_fn,
            "persistent_workers": config.num_workers > 0,
            "sampler": sampler,
            "shuffle": sampler is None,
        }

        tile_config = self.tile_config
        if tile_config.enable_tiler and tile_config.sampling_ratio < 1:
            num_samples = max(1, int(len(dataset) * tile_config.sampling_ratio))
            log.info(f"Using tiled sampling with {num_samples} samples")
            common_args.update(
                {
                    "shuffle": False,
                    "sampler": RandomSampler(dataset, num_samples=num_samples),
                },
            )
        return DataLoader(**common_args)

    def val_dataloader(self) -> DataLoader:
        """Get val dataloader."""
        if self.val_subset is None:
            self.val_subset = self._get_default_subset_config("val")

        config = self.val_subset
        dataset = self._get_dataset(config.subset_name)

        return DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            persistent_workers=config.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        if self.test_subset is None:
            # Use default configuration when datasets are provided directly
            self.test_subset = self._get_default_subset_config("test")

        config = self.test_subset
        dataset = self._get_dataset(config.subset_name)

        return DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            persistent_workers=config.num_workers > 0,
        )

    @property
    def default_input_size_per_task(self):
        """Return default input size per OTX task type."""
        task_to_size = {
            OTXTaskType.ANOMALY: (256, 256),
            OTXTaskType.ANOMALY_CLASSIFICATION: (256, 256),
            OTXTaskType.ANOMALY_DETECTION: (256, 256),
            OTXTaskType.ANOMALY_SEGMENTATION: (256, 256),
            OTXTaskType.MULTI_CLASS_CLS: (224, 224),
            OTXTaskType.MULTI_LABEL_CLS: (224, 224),
            OTXTaskType.H_LABEL_CLS: (224, 224),
            OTXTaskType.DETECTION: (640, 640),
            OTXTaskType.INSTANCE_SEGMENTATION: (640, 640),
            OTXTaskType.SEMANTIC_SEGMENTATION: (512, 512),
            OTXTaskType.ROTATED_DETECTION: (640, 640),
            OTXTaskType.KEYPOINT_DETECTION: (512, 512),
        }

        return task_to_size[self.task]

    def setup(self, stage: str) -> None:
        """Setup for each stage."""

    def teardown(self, stage: str) -> None:
        """Teardown for each stage."""
        # clean up after fit or test
        # called on every process in DDP

    @property
    def hparams_initial(self) -> AttributeDict:
        """The collection of hyperparameters saved with `save_hyperparameters()`. It is read-only.

        The reason why we override is that we have some custom resolvers for `DictConfig`.
        Some resolved Python objects has not a primitive type, so that is not loggable without errors.
        Therefore, we need to unresolve it this time.
        """
        hp = super().hparams_initial
        for key, value in hp.items():
            if isinstance(value, DictConfig):
                # It should be unresolved to make it loggable
                hp[key] = OmegaConf.to_container(value, resolve=False)

        return hp

    def __reduce__(self):
        """Re-initialize object when unpickled."""
        return (
            self.__class__,
            (
                self.task,
                self.data_format,
                self.data_root,
                self.train_subset,
                self.val_subset,
                self.test_subset,
                self.tile_config,
                self.image_color_channel,
                self.include_polygons,
                self.ignore_index,
                self.unannotated_items_ratio,
                self.auto_num_workers,
                self.device,
                self.input_size,
            ),
        )
        )
