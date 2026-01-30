# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import datumaro.experimental as dm
import polars as pl
from datumaro.experimental.export_import import export_dataset, import_dataset
from loguru import logger
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

from app.db.schema import DatasetRevisionDB
from app.models import DatasetItemSubset
from app.models.dataset_item_revision import DatasetRevisionItem
from app.models.dataset_revision import DatasetRevision
from app.repositories import DatasetRevisionRepository
from app.utils.images import crop_to_thumbnail

from .base import BaseSessionManagedService, ResourceNotFoundError, ResourceType
from .media_service import InvalidImageError


class DatasetRevisionService(BaseSessionManagedService):
    def __init__(self, data_dir: Path, db_session: Session | None = None) -> None:
        super().__init__(db_session)
        self.projects_dir = data_dir / "projects"

    def save_revision(self, project_id: UUID, dataset: dm.Dataset) -> UUID:
        """
        Saves the dataset as a new revision.

        Creates a new dataset revision entry in the database and exports the dataset
        to a zip file in the project's revisions directory.

        Args:
            project_id: The UUID of the project to save the revision for.
            dataset: The Datumaro dataset to export.

        Returns:
            UUID: The UUID of the newly created dataset revision.
        """
        revision_repo = DatasetRevisionRepository(project_id=str(project_id), db=self.db_session)
        dataset_revision_id = str(uuid4())
        short_id = dataset_revision_id.split("-")[0]
        dataset_name = f"Dataset ({short_id})"
        revision_db = revision_repo.save(
            DatasetRevisionDB(
                id=dataset_revision_id,
                project_id=str(project_id),
                name=dataset_name,
            )
        )
        revision_path = self.projects_dir / str(project_id) / "dataset_revisions" / revision_db.id
        logger.info("Saving dataset revision '{}' to '{}'", revision_db.id, revision_path)
        export_dataset(
            dataset=dataset,
            output_path=revision_path,
            export_images=True,
            as_zip=False,  # Export as uncompressed directory, see #5070 for details
        )
        return UUID(revision_db.id)

    def load_revision(self, project_id: UUID, dataset_revision_id: UUID) -> dm.Dataset:
        """
        Loads the Datumaro dataset belonging to the dataset revision.

        Args:
            project_id: The UUID of the project.
            dataset_revision_id: The UUID of the dataset revision.
        Returns:
            dm.Dataset: The dataset revision as a Datumaro dataset.
        """
        dataset_revision = self.get_dataset_revision(project_id, dataset_revision_id)
        if dataset_revision.files_deleted:
            raise ResourceNotFoundError(ResourceType.DATASET_REVISION, str(dataset_revision_id))
        parquet_path = self._get_revision_parquet_path(project_id, dataset_revision_id)
        return import_dataset(input_path=parquet_path.parent)

    def list_dataset_revisions(self, project_id: UUID) -> list[DatasetRevision]:
        """
        Get information about all available dataset revisions in a project.

        Retrieves a list of all dataset revisions that belong to the specified project.

        Args:
            project_id (UUID): The unique identifier of the project whose dataset revisions to list.

        Returns:
            list[DatasetRevision]: A list of dataset revision objects representing all dataset
                revisions in the project. Returns an empty list if the project has no dataset revisions.
        """
        dataset_revision_repo = DatasetRevisionRepository(project_id=str(project_id), db=self.db_session)
        return [DatasetRevision.model_validate(dataset_rev_db) for dataset_rev_db in dataset_revision_repo.list_all()]

    def get_dataset_revision(self, project_id: UUID, revision_id: UUID) -> DatasetRevision:
        """
        Get a dataset revision by ID.

        Args:
            project_id: The UUID of the project.
            revision_id: The UUID of the dataset revision.

        Returns:
            DatasetRevision: The dataset revision.

        Raises:
            ResourceNotFoundError: If the revision is not found.
        """
        revision_repo = DatasetRevisionRepository(project_id=str(project_id), db=self.db_session)
        revision = revision_repo.get_by_id(str(revision_id))
        if revision is None:
            raise ResourceNotFoundError(ResourceType.DATASET_REVISION, str(revision_id))
        return self._to_dataset_revision(dataset_db=revision)

    def rename_dataset_revision(
        self, project_id: UUID, dataset_revision: DatasetRevision, new_name: str
    ) -> DatasetRevision:
        """
        Rename a dataset revision.

        Args:
            project_id: The UUID of the project.
            dataset_revision: The dataset revision to rename.
            new_name: The new name to assign to the dataset revision.

        Returns:
            DatasetRevision: The dataset revision object containing the dataset revision's updated information.
        """
        if new_name is not None:
            dataset_revision.name = new_name
            self.update_dataset_revision(project_id=project_id, dataset_revision=dataset_revision)
        return dataset_revision

    def update_dataset_revision(self, project_id: UUID, dataset_revision: DatasetRevision) -> None:
        """
        Updates a dataset revision.

        Args:
            project_id: The UUID of the project.
            dataset_revision: The dataset revision to update.
        """
        revision_repo = DatasetRevisionRepository(project_id=str(project_id), db=self.db_session)
        _ = revision_repo.update(
            DatasetRevisionDB(
                id=str(dataset_revision.id),
                project_id=str(project_id),
                name=dataset_revision.name,
                files_deleted=dataset_revision.files_deleted,
            )
        )

    def delete_dataset_revision_files(self, project_id: UUID, revision_id: UUID) -> None:
        """
        Marks the DatasetRevision files as deleted, and deletes associated files from the disk.

        Args:
            project_id: The UUID of the project.
            revision_id: The UUID of the dataset revision.

        Raises:
            ResourceNotFoundError: If the revision is not found.
        """
        revision = self.get_dataset_revision(project_id, revision_id)
        if revision.files_deleted:
            logger.info("Files for dataset revision '{}' are already deleted", revision_id)
            return

        # Mark as deleted in the database
        revision.files_deleted = True
        self.update_dataset_revision(project_id=project_id, dataset_revision=revision)

        # Delete files from filesystem
        revision_path = self.projects_dir / str(project_id) / "dataset_revisions" / str(revision_id)
        if revision_path.exists():
            shutil.rmtree(revision_path)
            logger.info("Deleted dataset revision files at '{}'", revision_path)

    def _get_revision_parquet_path(self, project_id: UUID, revision_id: UUID) -> Path:
        """
        Get the path to the parquet file for a dataset revision.

        Args:
            project_id: The UUID of the project.
            revision_id: The UUID of the dataset revision.

        Returns:
            Path to the data.parquet file.

        Raises:
            ResourceNotFoundError: If the parquet file is not found.
        """
        parquet_path = self.projects_dir / str(project_id) / "dataset_revisions" / str(revision_id) / "data.parquet"
        if not parquet_path.exists():
            raise ResourceNotFoundError(ResourceType.DATASET_REVISION, str(revision_id))

        return parquet_path

    @lru_cache(maxsize=64)
    def count_items_by_subset(self, project_id: UUID, dataset_revision_id: UUID) -> dict[str, int]:
        """
        Count the number of dataset items in a dataset revision, grouped by subset.

        Efficiently reads only the necessary metadata from the Parquet file to compute:
        - The total number of items ("total")
        - The number of items in each subset (e.g., "train", "val", "test")

        Args:
            project_id (UUID): The UUID of the project.
            dataset_revision_id (UUID): The UUID of the dataset revision.

        Returns:
            dict[str, int]: A dictionary mapping subset names to counts. Includes a "total" key for the total count.

        Raises:
            ResourceNotFoundError: If the Parquet file does not exist.
        """
        parquet_path = self._get_revision_parquet_path(project_id, dataset_revision_id)
        lf = pl.scan_parquet(parquet_path)
        subset_counts = lf.group_by("subset").len().collect()
        counts = {row["subset"].lower(): row["len"] for row in subset_counts.to_dicts()}
        total = sum(counts.values())
        counts["total"] = total
        return counts

    def _parse_revision_item_from_df_row_dict(
        self, project_id: UUID, dataset_revision_id: UUID, df_revision_item: dict[str, Any]
    ) -> DatasetRevisionItem:
        """
        Utility method to convert a dataset revision item from a dict representation
        (a row of the Polars dataframe, serialized with to_dicts()) to a DatasetRevisionItem.

        The dict is expected to have the following structure:
        {
            "id": "9bedc1ec-edf1-4eec-8549-82239c585c48",
            "image": "relative/path/to/image.jpg",
            "image_info": {
                "width": 1024,
                "height": 768
            },
            "subset": "TRAINING",
            ... other fields ...
        }

        Args:
            project_id: The UUID of the project.
            dataset_revision_id: The UUID of the dataset revision.
            df_revision_item: The dataset revision item as a dict.

        Returns:
            DatasetRevisionItem
        """
        dataset_revision_path = self.projects_dir / str(project_id) / "dataset_revisions" / str(dataset_revision_id)
        try:
            # The path stored in the 'image' column is relative to the dataset revision base folder
            image_rel_path = Path(df_revision_item["image"])

            if "data/projects" in str(image_rel_path):
                # TODO due to a bug in Datumaro, the above assumption doesn't always hold true, and the image path
                #  may be relative to the app data folder instead. This workaround can be removed after the bug is fixed
                #  in Datumaro: https://github.com/open-edge-platform/datumaro/issues/2019
                image_path = image_rel_path
            else:
                image_path = dataset_revision_path / image_rel_path

            if "id" not in df_revision_item:
                # Fallback: if the 'id' field is missing, try to extract it from the image filename
                # TODO remove this fallback logic after ensuring the 'id' is always saved in the parquet file
                #  https://github.com/open-edge-platform/training_extensions/issues/5350
                logger.warning(
                    "Dataset revision item is missing 'id' field, attempting to extract from image filename: {}",
                    image_rel_path,
                )
                df_revision_item["id"] = image_rel_path.stem  # Filename without extension

            return DatasetRevisionItem(
                id=UUID(df_revision_item["id"]),
                format=image_rel_path.suffix.lstrip("."),
                image_path=image_path,
                width=df_revision_item["image_info"]["width"],
                height=df_revision_item["image_info"]["height"],
                subset=DatasetItemSubset(df_revision_item["subset"].lower()),
            )
        except Exception as e:
            logger.error(
                "Failed to parse dataset revision item from dataframe row dict: {}. Error: {}",
                df_revision_item,
                e,
            )
            raise

    def list_dataset_revision_items(
        self,
        project_id: UUID,
        dataset_revision: DatasetRevision,
        limit: int = 10,
        offset: int = 0,
        subset: DatasetItemSubset | None = None,
    ) -> tuple[list[DatasetRevisionItem], int]:
        """
        List items in a dataset revision with pagination and filtering.

        Args:
            project_id: The UUID of the project.
            dataset_revision: The dataset revision.
            limit: Maximum number of items to return.
            offset: Number of items to skip.
            subset: Optional subset filter.

        Returns:
            Tuple of (list of items as dicts, total count).
        """
        parquet_path = self._get_revision_parquet_path(project_id, dataset_revision.id)

        df = pl.scan_parquet(parquet_path)

        if subset is not None:
            df = df.filter(pl.col("subset") == subset.name)

        total_count = df.select(pl.len()).collect().item()
        df = df.slice(offset, limit).collect()

        dataset_revision_items = []
        for item in df.to_dicts():
            dataset_revision_items.append(
                self._parse_revision_item_from_df_row_dict(
                    project_id=project_id, dataset_revision_id=dataset_revision.id, df_revision_item=item
                )
            )
        return dataset_revision_items, total_count

    def get_dataset_revision_item(
        self,
        project_id: UUID,
        dataset_revision: DatasetRevision,
        item_id: str,
    ) -> DatasetRevisionItem:
        """
        Get a specific item from a dataset revision by ID.

        Args:
            project_id: The UUID of the project.
            dataset_revision: The dataset revision.
            item_id: The ID of the item within the dataset.

        Returns:
            DatasetRevisionItem: The requested dataset revision item.

        Raises:
            ResourceNotFoundError: If the item is not found.
        """
        parquet_path = self._get_revision_parquet_path(project_id, dataset_revision.id)

        df = pl.scan_parquet(parquet_path)

        if "id" in df.columns:
            df = df.filter(pl.col("id") == item_id).collect()
        else:  # Fallback: locate the item through the 'image' column, if it contains the item ID
            logger.warning("Dataset revision parquet does not contain 'id' column, falling back to search via 'image'")
            df = df.filter(pl.col("image").str.contains(item_id)).collect()
        if len(df) == 0:
            raise ResourceNotFoundError(ResourceType.DATASET_ITEM, item_id)

        item = df.to_dicts()[0]
        item["id"] = item_id  # Ensure the ID is set correctly

        return self._parse_revision_item_from_df_row_dict(
            project_id=project_id, dataset_revision_id=dataset_revision.id, df_revision_item=item
        )

    def get_dataset_revision_item_thumbnail(
        self,
        project_id: UUID,
        dataset_revision: DatasetRevision,
        item_id: str,
    ) -> Image.Image:
        """
        Generate and return a thumbnail for a specific dataset revision item.

        Args:
            project_id: The UUID of the project.
            dataset_revision: The dataset revision.
            item_id: The ID of the item within the dataset.

        Returns:
            Image.Image: The generated thumbnail image.
        """
        binary_path = self.get_dataset_revision_item(
            project_id=project_id,
            dataset_revision=dataset_revision,
            item_id=item_id,
        ).image_path
        try:
            with Image.open(binary_path) as image:
                thumbnail = crop_to_thumbnail(image=image, target_width=64, target_height=64)
        except UnidentifiedImageError:
            logger.error("Failed to open image {} for thumbnail generation", binary_path)
            raise InvalidImageError("Failed to open image for thumbnail generation.")
        return thumbnail

    @staticmethod
    def _to_dataset_revision(dataset_db: DatasetRevisionDB) -> DatasetRevision:
        """Convert database model to DatasetRevision."""
        return DatasetRevision.model_validate(dataset_db, from_attributes=True)
