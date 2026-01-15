# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path
from uuid import UUID

import datumaro.experimental as dm
import polars as pl
from datumaro.experimental.export_import import export_dataset
from loguru import logger
from sqlalchemy.orm import Session

from app.db.schema import DatasetRevisionDB
from app.models import DatasetItemSubset
from app.models.dataset_item_revision import DatasetRevisionItem
from app.models.dataset_revision import DatasetRevision
from app.repositories import DatasetRevisionRepository

from .base import BaseSessionManagedService, ResourceNotFoundError, ResourceType


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
        revision_repo = DatasetRevisionRepository(db=self.db_session)
        revision_db = revision_repo.save(
            DatasetRevisionDB(
                project_id=str(project_id),
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
        revision_repo = DatasetRevisionRepository(db=self.db_session)
        revision = revision_repo.get_by_id(str(revision_id))
        if revision is None or revision.project_id != str(project_id):
            raise ResourceNotFoundError(ResourceType.DATASET_REVISION, str(revision_id))
        return self._to_dataset_revision(dataset_db=revision)

    def update_dataset_revision(self, dataset_revision: DatasetRevision) -> None:
        """
        Updates a dataset revision.

        Args:
            dataset_revision: The dataset revision to update.
        """
        revision_repo = DatasetRevisionRepository(db=self.db_session)
        _ = revision_repo.update(
            DatasetRevisionDB(
                id=str(dataset_revision.id),
                project_id=str(dataset_revision.project_id),
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
        self.update_dataset_revision(dataset_revision=revision)

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

        df = pl.read_parquet(parquet_path)

        if subset is not None:
            df = df.filter(pl.col("subset") == subset.name)

        total_count = len(df)
        df = df.slice(offset, limit)

        dataset_revision_items = []
        for item in df.to_dicts():
            dataset_revision_items.append(
                DatasetRevisionItem.model_validate(
                    {
                        "id": item["id"],
                        "name": Path(item["image"]).stem,
                        "format": Path(item["image"]).suffix.lstrip("."),
                        "width": item["image_info"]["width"],
                        "height": item["image_info"]["height"],
                        "subset": item["subset"],
                    }
                )
            )
        return dataset_revision_items, total_count

    def get_dataset_revision_item(
        self,
        project_id: UUID,
        dataset_revision: DatasetRevision,
        item_id: str,
    ) -> dict:
        """
        Get a specific item from a dataset revision by ID.

        Args:
            project_id: The UUID of the project.
            dataset_revision: The dataset revision.
            item_id: The ID of the item within the dataset.

        Returns:
            Dictionary containing the item data.

        Raises:
            ResourceNotFoundError: If the item is not found.
        """
        parquet_path = self._get_revision_parquet_path(project_id, dataset_revision.id)

        df = pl.read_parquet(parquet_path)

        if "id" in df.columns:
            filtered = df.filter(pl.col("id") == item_id)
        else:
            # Fallback: try to use image column
            filtered = df.filter(pl.col("image").str.contains(item_id))

        if len(filtered) == 0:
            raise ResourceNotFoundError(ResourceType.DATASET_ITEM, item_id)

        return filtered.to_dicts()[0]

    def get_dataset_revision_item_binary_path(
        self,
        project_id: UUID,
        dataset_revision: DatasetRevision,
        item_id: str,
    ) -> Path:
        """
        Get the path to the binary file for a dataset revision item.

        Args:
            project_id: The UUID of the project.
            dataset_revision: The dataset revision.
            item_id: The ID of the item within the dataset.

        Returns:
            Path to the image file.
        """
        item = self.get_dataset_revision_item(project_id, dataset_revision, item_id)
        revision_path = self.projects_dir / str(project_id) / "dataset_revisions" / str(dataset_revision.id)

        if isinstance(item.get("image"), str):
            image_path = revision_path / "images" / Path(item["image"]).name
        else:
            # Fallback: try to find by item_id
            images_dir = revision_path / "images"
            for file in images_dir.glob(f"*{item_id}*"):
                return file
            raise ResourceNotFoundError(ResourceType.DATASET_ITEM, item_id)

        return image_path

    @staticmethod
    def _to_dataset_revision(dataset_db: DatasetRevisionDB) -> DatasetRevision:
        """Convert database model to DatasetRevision."""
        return DatasetRevision.model_validate(dataset_db, from_attributes=True)
