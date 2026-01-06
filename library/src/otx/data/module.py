# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LightningDataModule extension for OTX."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from datumaro import Dataset as DmDataset
from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, RandomSampler
from torchvision.transforms.v2 import Normalize

from otx.config.data import SubsetConfig, TileConfig
from otx.data.dataset.tile import OTXTileDatasetFactory
from otx.data.factory import OTXDatasetFactory
from otx.data.transform_libs.torchvision import Compose, TorchVisionTransformLib
from otx.data.utils import adapt_tile_config, get_adaptive_num_workers, instantiate_sampler
from otx.data.utils.pre_filtering import pre_filtering
from otx.types.device import DeviceType
from otx.types.label import LabelInfo
from otx.types.task import OTXTaskType

if TYPE_CHECKING:
    from lightning.pytorch.utilities.parsing import AttributeDict

    from otx.data.dataset.base import OTXDataset

logger = logging.getLogger(__name__)


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

        if input_size is not None and not isinstance(input_size, (tuple, list)):
            msg = f"input_size should be a tuple or list of ints, but got {input_size!r}"
            raise ValueError(msg)

        subset_configs = self.get_default_subset_configs(input_size)
        self.train_subset = train_subset if train_subset is not None else subset_configs["train_subset"]
        self.val_subset = val_subset if val_subset is not None else subset_configs["val_subset"]
        self.test_subset = test_subset if test_subset is not None else subset_configs["test_subset"]
        self.tile_config = tile_config

        self.ignore_index = ignore_index
        self.unannotated_items_ratio = unannotated_items_ratio

        self.auto_num_workers = auto_num_workers
        self.device = device

        self.subsets: dict[str, OTXDataset] = {}
        self.save_hyperparameters(ignore=["input_size"])

        dataset = DmDataset.import_from(self.data_root, format=self.data_format)

        if self.data_format is None:
            self.data_format = dataset.format

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

        if input_size is not None:
            # override input_size to all subset configs when it is given
            for subset_cfg in [self.train_subset, self.val_subset, self.test_subset]:
                if subset_cfg.input_size is None:
                    subset_cfg.input_size = input_size  # type: ignore[assignment]

        # Extract mean and std from Normalize transform
        self.input_mean, self.input_std = self.extract_normalization_params(self.train_subset.transforms)
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
                logger.warning(
                    "Only GPU device type support auto_num_workers. "
                    f"Current deveice type is {self.device!s}. auto_num_workers is skipped.",
                )
            elif (num_workers := get_adaptive_num_workers()) is not None:
                for subset_name, subset_config in config_mapping.items():
                    logger.info(
                        f"num_workers of {subset_name} subset is changed : "
                        f"{subset_config.num_workers} -> {num_workers}",
                    )
                    subset_config.num_workers = num_workers

        label_infos: list[LabelInfo] = []
        for name, dm_subset in dataset.subsets().items():
            if name not in config_mapping:
                logger.warning(f"{name} is not available. Skip it")
                continue

            otx_dataset = OTXDatasetFactory.create(
                task=self.task,
                dm_subset=dm_subset.as_dataset(),
                cfg_subset=config_mapping[name],
                data_format=self.data_format,  # type: ignore[arg-type]
                ignore_index=self.ignore_index,
            )

            if self.tile_config.enable_tiler:
                otx_dataset = OTXTileDatasetFactory.create(
                    dataset=otx_dataset,
                    tile_config=self.tile_config,
                )
            self.subsets[name] = otx_dataset
            label_infos += [self.subsets[name].label_info]
            logger.info(f"Add name: {name}, self.subsets: {self.subsets}")

        if self._is_meta_info_valid(label_infos) is False:
            msg = "All data meta infos of subsets should be the same."
            raise ValueError(msg)

        self.label_info = next(iter(label_infos))

    @classmethod
    def extract_normalization_params(
        cls, transforms_source: Sequence[dict[str, Any]] | Compose | None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Extract mean and std from the dataset transforms.

        Specifically, this method looks for a Normalize transform in the provided transforms, and extracts
        the mean and std values used for normalization.
        If not found, it returns default values of mean=(0.0, 0.0, 0.0) and std=(1.0, 1.0, 1.0).

        Args:
            transforms_source: Transforms applied to the dataset.
                Should be specified as an iterable of transform descriptors (jsonargparse-like) or a Compose object

        Returns:
            Tuple of (mean, std) tuples.
        """
        mean = (0.0, 0.0, 0.0)
        std = (1.0, 1.0, 1.0)

        if transforms_source is None:
            return mean, std
        if hasattr(transforms_source, "__iter__"):
            transforms_iterable = transforms_source
        elif isinstance(transforms_source, Compose):
            transforms_iterable = transforms_source.transforms
        else:
            msg = f"Transforms should be given as an iterable or a Compose object, got {type(transforms_source)}"
            raise TypeError(msg)

        for transform in transforms_iterable:
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

        if len(mean) != 3 or len(std) != 3:
            msg = f"Expected mean and std to have length 3, got mean={mean}, std={std}"
            raise ValueError(msg)

        return tuple(mean), tuple(std)  # type: ignore[return-value]

    @classmethod
    def from_otx_datasets(
        cls,
        train_dataset: OTXDataset,
        val_dataset: OTXDataset,
        test_dataset: OTXDataset | None = None,
        train_subset: SubsetConfig | None = None,
        val_subset: SubsetConfig | None = None,
        test_subset: SubsetConfig | None = None,
        auto_num_workers: bool = False,
        device: DeviceType = DeviceType.auto,
    ) -> OTXDataModule:
        """Create an OTXDataModule from pre-constructed OTXDataset instances.

        This is a factory method that provides a clean way to create OTXDataModule instances
        when you already have constructed datasets, without needing to provide data_root,
        data_format, or other data loading parameters.

        Args:
            train_dataset (OTXDataset): Pre-constructed training dataset.
            val_dataset (OTXDataset): Pre-constructed validation dataset.
            test_dataset (OTXDataset | None, optional): Pre-constructed test dataset. Defaults to None.
            train_subset (SubsetConfig | None, optional): Configuration for the training dataloader.
                If None, default configuration will be used. Defaults to None.
                Transforms should be left unspecified as they will be taken from the provided train_dataset.
            val_subset (SubsetConfig | None, optional): Configuration for the validation dataloader.
                If None, default configuration will be used. Defaults to None.
                Transforms should be left unspecified as they will be taken from the provided val_dataset.
            test_subset (SubsetConfig | None, optional): Configuration for the test dataloader.
                If None, default configuration will be used. Defaults to None.
                Transforms should be left unspecified as they will be taken from the provided test_dataset.
            auto_num_workers (bool, optional): Whether to automatically determine the number of workers.
                Defaults to False.
            device (DeviceType, optional): Device type to use (e.g., 'cpu', 'gpu').
                Defaults to DeviceType.auto.

        Returns:
            OTXDataModule: Configured data module with the provided datasets.

        Raises:
            ValueError: If datasets dictionary is empty or datasets have inconsistent metadata.

        Examples:
            >>> from otx.data.module import OTXDataModule
            >>> from otx.types.task import OTXTaskType
            >>>
            >>> # Create datamodule with minimal configuration
            >>> datamodule = OTXDataModule.from_otx_datasets(
            ...     train_dataset=my_train_dataset,
            ...     val_dataset=my_val_dataset,
            ...     test_dataset=my_test_dataset,
            ... )
            >>>
            >>> # Create datamodule with custom subset configurations
            >>> from otx.config.data import SubsetConfig
            >>> train_config = SubsetConfig(
            ...     batch_size=64,
            ...     num_workers=8,
            ... )
            >>> datamodule = OTXDataModule.from_otx_datasets(
            ...     train_dataset=my_train_dataset,
            ...     val_dataset=my_val_dataset,
            ...     test_dataset=my_test_dataset,
            ...     train_subset=train_config,
            ... )
        """
        # Validate label info consistency across datasets
        if test_dataset is None:
            test_dataset = val_dataset

        if not all(
            label_info == train_dataset.label_info for label_info in [val_dataset.label_info, test_dataset.label_info]
        ):
            msg = "All data meta infos of provided datasets should be the same."
            raise ValueError(msg)

        # Create instance
        instance = cls.__new__(cls)
        LightningDataModule.__init__(instance)
        # Set basic attributes
        instance.subsets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
        instance.task = train_dataset.task_type  # type: ignore[assignment]
        instance.data_format = train_dataset.data_format
        instance.data_root = ""
        instance.tile_config = (
            train_dataset.tile_config if hasattr(train_dataset, "tile_config") else TileConfig(enable_tiler=False)
        )
        instance.ignore_index = 255
        instance.unannotated_items_ratio = 0.0
        instance.auto_num_workers = auto_num_workers
        instance.device = device

        # Store datasets and label info
        instance.label_info = train_dataset.label_info

        # Derive the image size from the dataset, assuming layout CHW and that all dataset items have
        # the same size after transforms.
        try:
            example_item = next(iter(train_dataset))
        except StopIteration:
            msg = "train_dataset is empty; cannot infer input_size"
            raise ValueError(msg) from None
        instance.input_size = example_item.image.shape[1:]

        # merge default configs with provided subsets
        default_subset_configs: dict[str, SubsetConfig] = {}  # Initialize lazily if needed)
        for name, subset in zip(["train", "val", "test"], [train_subset, val_subset, test_subset]):
            if subset is not None:
                # Use provided subset config
                subset_to_assign = subset
                if subset.transforms:
                    logger.warning(
                        f"The provided {name} SubsetConfig contains transforms which will be overridden "
                        "by the transforms of the provided OTXDataset. When building OTXDataModule from "
                        "pre-constructed datasets, developers should set up the transforms when creating the datasets.",
                    )
            else:
                # Use default config but get transforms from the dataset
                if not default_subset_configs:
                    default_subset_configs = instance.get_default_subset_configs(instance.input_size)
                subset_to_assign = default_subset_configs[f"{name}_subset"]

            # Override transforms with the ones from the pre-constructed dataset
            subset_to_assign.transforms = instance.subsets[name].transforms  # type: ignore[assignment]

            # Set the 'train_subset', 'val_subset', 'test_subset' attributes
            setattr(instance, f"{name}_subset", subset_to_assign)

        # Extract normalization parameters from train dataset transforms if available
        instance.input_mean, instance.input_std = instance.extract_normalization_params(
            instance.train_subset.transforms
        )

        # Save hyperparameters
        instance.save_hyperparameters(
            ignore=["subsets", "input_size"],
            logger=False,
        )

        return instance

    def get_default_subset_configs(self, input_size: tuple[int, int] | None = None) -> dict[str, SubsetConfig]:
        """Create a default SubsetConfig for a given subset when not provided.

        This method loads the configuration from the base YAML files in
        otx/recipe/_base_/data based on the task type.

        Args:
            input_size: input size of the image to set in subset configs.

        Returns:
            dict[str, SubsetConfig]: Default configuration for the subsets loaded from base config.
        """
        # Map task type to config file name
        task_to_data_config_file = {
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
        subset_dicts = {}
        for subset_key in ["train_subset", "val_subset", "test_subset"]:
            if subset_key not in config_dict:
                msg = f"Subset '{subset_key}' not found in config file {config_file}"
                raise ValueError(msg)

            # Extract subset config and convert to container (dict)
            subset_config_dict = OmegaConf.to_container(config_dict[subset_key], resolve=True)  # type: ignore[index]
            subset_input_size = subset_config_dict.get("input_size")
            if subset_input_size is None and input_size is not None:
                subset_config_dict["input_size"] = input_size
            elif subset_input_size is None and input_size is None:
                subset_config_dict["input_size"] = config_dict.get("input_size")

            if subset_config_dict["input_size"] is None:
                msg = "input size is not specified in both the config file and the DataModule constructor."
                raise ValueError(msg)
            subset_config_dict = SubsetConfig(**subset_config_dict)
            subset_config_dict.transforms = TorchVisionTransformLib.generate(subset_config_dict)
            subset_dicts[subset_key] = subset_config_dict
        return subset_dicts

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
            logger.info(f"Using tiled sampling with {num_samples} samples")
            common_args.update(
                {
                    "shuffle": False,
                    "sampler": RandomSampler(dataset, num_samples=num_samples),
                },
            )
        return DataLoader(**common_args)

    def val_dataloader(self) -> DataLoader:
        """Get val dataloader."""
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

    def predict_dataloader(self) -> DataLoader:
        """Get predict dataloader."""
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
                self.ignore_index,
                self.unannotated_items_ratio,
                self.auto_num_workers,
                self.device,
                self.input_size,
            ),
        )
