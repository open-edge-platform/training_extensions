# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import shutil
import tempfile
from pathlib import Path
from uuid import UUID
from zipfile import ZipFile

import datumaro.experimental as dm
import polars as pl
from datumaro.experimental.export_import import export_dataset
from loguru import logger
from sqlalchemy.orm import Session

from app.db.schema import DatasetRevisionDB
from app.models import DatasetItemSubset
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
            as_zip=True,
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

    def delete_dataset_revision_files(self, project_id: UUID, revision_id: UUID) -> None:
        """
        Delete the files associated with a dataset revision.

        Args:
            project_id: The UUID of the project.
            revision_id: The UUID of the dataset revision.

        Raises:
            ResourceNotFoundError: If the revision is not found.
        """
        revision = self.get_dataset_revision(project_id, revision_id)
        if revision.files_deleted:
            logger.info("Files for dataset revision '{}' already deleted", revision_id)
            return

        revision_path = self.projects_dir / str(project_id) / "dataset_revisions" / str(revision_id)
        if revision_path.exists():
            shutil.rmtree(revision_path)
            logger.info("Deleted dataset revision files at '{}'", revision_path)

        # Mark as deleted in the database
        revision_repo = DatasetRevisionRepository(db=self.db_session)
        revision_db = revision_repo.get_by_id(str(revision_id))
        if revision_db:
            revision_db.files_deleted = True
            revision_repo.save(revision_db)

    def _get_revision_parquet_path(self, project_id: UUID, revision_id: UUID) -> tuple[Path, bool]:
        """
        Get the path to the parquet file for a dataset revision.

        If the revision is stored as a zip, extract it temporarily to access the parquet file.

        Args:
            project_id: The UUID of the project.
            revision_id: The UUID of the dataset revision.

        Returns:
            Path to the data.parquet file (may be in a temp directory if extracted from zip).
            Bool indicating if the file was extracted from a zip and is in a temp directory.
        """
        revision_path = self.projects_dir / str(project_id) / "dataset_revisions" / str(revision_id)

        # When as_zip=True, datumaro creates dataset.zip inside the output directory
        zip_path = revision_path / "dataset.zip"
        if zip_path.exists():
            # Extract to temp directory
            temp_dir = Path(tempfile.mkdtemp(prefix=f"revision_{revision_id}_"))
            with ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
            return temp_dir / "data.parquet", True

        # Otherwise it's uncompressed (though this shouldn't happen with current implementation)
        return revision_path / "data.parquet", False

    def list_dataset_revision_items(
        self,
        project_id: UUID,
        revision_id: UUID,
        limit: int = 10,
        offset: int = 0,
        subset: DatasetItemSubset | None = None,
    ) -> tuple[list[dict], int]:
        """
        List items in a dataset revision with pagination and filtering.

        Args:
            project_id: The UUID of the project.
            revision_id: The UUID of the dataset revision.
            limit: Maximum number of items to return.
            offset: Number of items to skip.
            subset: Optional subset filter.

        Returns:
            Tuple of (list of items as dicts, total count).
        """
        parquet_path, is_temp = self._get_revision_parquet_path(project_id, revision_id)

        if not parquet_path.exists():
            raise ResourceNotFoundError(ResourceType.DATASET_REVISION, str(revision_id))

        try:
            df = pl.read_parquet(parquet_path)

            if subset is not None:
                df = df.filter(pl.col("subset") == subset.name)

            total_count = len(df)
            df = df.slice(offset, limit)
            items = df.to_dicts()
            return items, total_count
        finally:
            if is_temp:  # Clean up temp directory if we extracted a zip
                shutil.rmtree(parquet_path.parent, ignore_errors=True)

    def get_dataset_revision_item(
        self,
        project_id: UUID,
        revision_id: UUID,
        item_id: str,
    ) -> dict:
        """
        Get a specific item from a dataset revision by ID.

        Args:
            project_id: The UUID of the project.
            revision_id: The UUID of the dataset revision.
            item_id: The ID of the item within the dataset.

        Returns:
            Dictionary containing the item data.

        Raises:
            ResourceNotFoundError: If the item is not found.
        """
        parquet_path, is_temp = self._get_revision_parquet_path(project_id, revision_id)

        if not parquet_path.exists():
            raise ResourceNotFoundError(ResourceType.DATASET_REVISION, str(revision_id))

        try:
            df = pl.read_parquet(parquet_path)

            if "id" in df.columns:
                filtered = df.filter(pl.col("id") == item_id)
            else:
                # Fallback: try to use image column
                filtered = df.filter(pl.col("image").str.contains(item_id))

            if len(filtered) == 0:
                raise ResourceNotFoundError(ResourceType.DATASET_ITEM, item_id)

            return filtered.to_dicts()[0]
        finally:
            if is_temp:  # Clean up temp directory if we extracted a zip
                shutil.rmtree(parquet_path.parent, ignore_errors=True)

    def get_dataset_revision_item_binary_path(
        self,
        project_id: UUID,
        revision_id: UUID,
        item_id: str,
    ) -> Path:
        """
        Get the path to the binary file for a dataset revision item.

        Args:
            project_id: The UUID of the project.
            revision_id: The UUID of the dataset revision.
            item_id: The ID of the item within the dataset.

        Returns:
            Path to the image file.
        """
        item = self.get_dataset_revision_item(project_id, revision_id, item_id)

        revision_path = self.projects_dir / str(project_id) / "dataset_revisions" / str(revision_id)

        # When as_zip=True, datumaro creates dataset.zip inside the output directory
        zip_path = revision_path / "dataset.zip"
        if zip_path.exists():
            # For zip files, we need to extract to temp directory
            temp_dir = Path(tempfile.mkdtemp(prefix=f"revision_{revision_id}_item_"))
            with ZipFile(zip_path, "r") as zip_ref:
                # Extract only the specific image file
                image_rel_path = Path(item["image"]).name if isinstance(item.get("image"), str) else f"{item_id}.jpg"
                image_path_in_zip = f"images/{image_rel_path}"
                try:
                    zip_ref.extract(image_path_in_zip, temp_dir)
                    return temp_dir / image_path_in_zip
                except KeyError:
                    logger.warning(
                        "Unexpected archive structure in {}: could not find {}. "
                        "Searching for image file by item_id or image name.",
                        zip_path,
                        image_path_in_zip,
                    )
                    # Try alternative path
                    zip_ref.extractall(temp_dir)
                    # Find the image file
                    for file in (temp_dir / "images").glob("*"):
                        if item_id in file.name or (
                            isinstance(item.get("image"), str) and Path(item["image"]).name == file.name
                        ):
                            return file
                    raise ResourceNotFoundError(ResourceType.DATASET_ITEM, item_id)

        # For uncompressed, construct the path directly
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
