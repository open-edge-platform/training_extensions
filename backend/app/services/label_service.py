# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from sqlalchemy.exc import IntegrityError

from app.db import get_db_session
from app.db.schema import LabelDB
from app.repositories import LabelRepository
from app.schemas import Label
from app.services import ResourceAlreadyExistsError, ResourceType
from app.services.mappers.label_mapper import LabelMapper


def _convert_labels_to_db(labels: list[Label], project_id: UUID) -> list[LabelDB]:
    db_labels: list[LabelDB] = []
    for label in labels:
        db_label = LabelMapper.from_schema(label)
        db_label.project_id = str(project_id)
        db_labels.append(db_label)
    return db_labels


class LabelService:
    @staticmethod
    def update_labels_in_project(
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
            ResourceAlreadyExistsError: If any label to be added or updated would
                                       violate uniqueness constraints (same name,
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
            with get_db_session() as db:
                label_repo = LabelRepository(project_id=str(project_id), db=db)
                if labels_to_update:
                    label_repo.update_batch(_convert_labels_to_db(labels_to_update, project_id))
                if label_ids_to_remove:
                    label_repo.delete_batch([str(lid) for lid in label_ids_to_remove])
                if labels_to_add:
                    label_repo.save_batch(_convert_labels_to_db(labels_to_add, project_id))
                db.commit()
                label_dbs = label_repo.list_all()
                return [LabelMapper.to_schema(label_db) for label_db in label_dbs]
        except IntegrityError as e:
            if "unique constraint failed" in str(e).lower():
                raise ResourceAlreadyExistsError(
                    ResourceType.LABEL,
                    "",
                    message="Label with the same name or hotkey already exists in this project.",
                )
            raise
