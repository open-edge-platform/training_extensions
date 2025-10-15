# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from sqlalchemy.orm import Session

from app.db.schema import LabelDB
from app.repositories import LabelRepository
from app.repositories.base import PrimaryKeyIntegrityError, UniqueConstraintIntegrityError
from app.schemas import Label

from .base import ResourceType, ResourceWithIdAlreadyExistsError
from .mappers.label_mapper import LabelMapper


def _convert_labels_to_db(labels: list[Label], project_id: UUID) -> list[LabelDB]:
    db_labels: list[LabelDB] = []
    for label in labels:
        db_label = LabelMapper.from_schema(project_id=project_id, label=label)
        db_label.project_id = str(project_id)
        db_labels.append(db_label)
    return db_labels


class DuplicateLabelsError(Exception):
    """Exception raised when label with duplicated names or hotkeys are being stored."""

    def __init__(self):
        super().__init__("Either label names or hotkeys have duplicates")


class LabelService:
    def __init__(self, db_session: Session):
        self._db_session = db_session

    def create_label(self, project_id: UUID, label: Label) -> Label:
        label_repo = LabelRepository(str(project_id), self._db_session)
        try:
            saved = label_repo.save(LabelMapper.from_schema(project_id=project_id, label=label))
            return LabelMapper.to_schema(saved)
        except UniqueConstraintIntegrityError:
            raise DuplicateLabelsError
        except PrimaryKeyIntegrityError:
            raise ResourceWithIdAlreadyExistsError(ResourceType.LABEL, str(label.id))

    def list_all(self, project_id: UUID) -> list[Label]:
        label_repo = LabelRepository(str(project_id), self._db_session)
        labels = label_repo.list_all()
        return [LabelMapper.to_schema(label) for label in labels]

    def update_labels_in_project(
        self,
        project_id: UUID,
        labels_to_add: list[Label] | None,
        labels_to_update: list[Label] | None,
        label_ids_to_remove: list[UUID] | None,
    ) -> list[Label]:
        """
        Add, update, and remove labels in a project transactionally.

        This method performs label operations in a specific, deterministic order:
        1. First updates existing labels
        2. Then removes labels (by their IDs)
        3. Finally adds new labels

        This order ensures data integrity and prevents conflicts when operations
        might affect the same labels in different ways.

        Args:
            project_id: The UUID of the project to modify labels for
            labels_to_add: List of new Label objects to create. Each label must have
                          a unique name, hotkey, and color within the project.
            labels_to_update: List of existing Label objects with updated properties.
                             Labels must exist in the project and maintain uniqueness
                             constraints after modification.
            label_ids_to_remove: List of label UUIDs to delete from the project.

        Returns:
            list[Label]: The complete list of all labels in the project after all
                        operations have been applied, in the order returned by
                        the database.

        Raises:
            DuplicateLabelsError: If any label to be added or updated would violate uniqueness constraints (same name,
                                       or hotkey as an existing label in the project).
            IntegrityError: For other database integrity violations.
            DatabaseError: For general database operation failures.

        Example:
            >>> # Remove some labels, update others, and add new ones
            >>> updated_labels = update_labels_in_project(
            ...     project_id=project_uuid,
            ...     label_ids_to_remove=[uuid1, uuid2],
            ...     labels_to_update=[updated_label1, updated_label2],
            ...     labels_to_add=[new_label1, new_label2]
            ... )

        Note:
            The operation is atomic - either all changes succeed or none are applied.
            Uniqueness constraints are enforced at the database level for name and hotkey within each project.
        """
        try:
            label_repo = LabelRepository(project_id=str(project_id), db=self._db_session)
            if labels_to_update:
                label_repo.update_batch(_convert_labels_to_db(labels_to_update, project_id))
            if label_ids_to_remove:
                label_repo.delete_batch([str(lid) for lid in label_ids_to_remove])
            if labels_to_add:
                label_repo.save_batch(_convert_labels_to_db(labels_to_add, project_id))
            label_dbs = label_repo.list_all()
            return [LabelMapper.to_schema(label_db) for label_db in label_dbs]
        except UniqueConstraintIntegrityError:
            raise DuplicateLabelsError
