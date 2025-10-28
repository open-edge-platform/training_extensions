# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from sqlalchemy.orm import Session

from app.db.schema import LabelDB
from app.repositories import LabelRepository
from app.repositories.base import PrimaryKeyIntegrityError, UniqueConstraintIntegrityError
from app.schemas import LabelCreate, LabelView
from app.schemas.label import LabelEdit, LabelRemove
from app.utils.color import random_color

from .base import ResourceType, ResourceWithIdAlreadyExistsError
from .mappers.label_mapper import LabelMapper


class DuplicateLabelsError(Exception):
    """Exception raised when label with duplicated names or hotkeys are being stored."""

    def __init__(self):
        super().__init__("Either label names or hotkeys have duplicates")


class LabelService:
    def __init__(self, db_session: Session):
        self._db_session = db_session

    def create_label(self, project_id: UUID, label: LabelCreate) -> LabelView:
        label_repo = LabelRepository(str(project_id), self._db_session)
        try:
            saved = label_repo.save(
                LabelDB(
                    id=str(label.id) if label.id is not None else None,
                    project_id=str(project_id),
                    name=label.name,
                    color=label.color if label.color is not None else random_color(),
                    hotkey=label.hotkey,
                )
            )
            return LabelMapper.to_schema(saved)
        except UniqueConstraintIntegrityError:
            raise DuplicateLabelsError
        except PrimaryKeyIntegrityError:
            raise ResourceWithIdAlreadyExistsError(ResourceType.LABEL, str(label.id))

    def list_all(self, project_id: UUID) -> list[LabelView]:
        label_repo = LabelRepository(str(project_id), self._db_session)
        labels = label_repo.list_all()
        return [LabelMapper.to_schema(label) for label in labels]

    def update_labels_in_project(
        self,
        project_id: UUID,
        labels_to_add: list[LabelCreate] | None,
        labels_to_update: list[LabelEdit] | None,
        labels_to_remove: list[LabelRemove] | None,
    ) -> list[LabelView]:
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
            labels_to_remove: List of labels to delete from the project.

        Returns:
            list[LabelView]: The complete list of all labels in the project after all
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
            ...     labels_to_remove=[deleted_label1, deleted_label2],
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
                batch = [
                    LabelDB(
                        id=str(label.id),
                        project_id=str(project_id),
                        name=label.new_name,
                        color=label.new_color,
                        hotkey=label.new_hotkey,
                    )
                    for label in labels_to_update
                ]
                label_repo.update_batch(batch)
            if labels_to_remove:
                label_repo.delete_batch([str(label.id) for label in labels_to_remove])
            if labels_to_add:
                batch = [
                    LabelDB(
                        project_id=str(project_id),
                        name=label.name,
                        color=label.color if label.color is not None else random_color(),
                        hotkey=label.hotkey,
                    )
                    for label in labels_to_add
                ]
                label_repo.save_batch(batch)
            label_dbs = label_repo.list_all()
            return [LabelMapper.to_schema(label_db) for label_db in label_dbs]
        except UniqueConstraintIntegrityError:
            raise DuplicateLabelsError
