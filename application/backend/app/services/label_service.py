# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from sqlalchemy.orm import Session

from app.db.schema import LabelDB
from app.models import Label
from app.models.label import LabelReference, LabelUpdateInfo
from app.models.project import Project
from app.models.task import TaskType
from app.repositories import LabelRepository
from app.repositories.base import PrimaryKeyIntegrityError, UniqueConstraintIntegrityError
from app.utils.color import random_color

from . import BaseSessionManagedService
from .base import ResourceNotFoundError, ResourceType, ResourceWithIdAlreadyExistsError


class DuplicateLabelsError(Exception):
    """Exception raised when label with duplicated names or hotkeys are being stored."""

    def __init__(self):
        super().__init__("Either label names or hotkeys have duplicates")


class LabelService(BaseSessionManagedService):
    def __init__(self, db_session: Session | None = None):
        super().__init__(db_session)

    def create_label(
        self, project_id: UUID, name: str, color: str | None, hotkey: str | None, label_id: UUID | None = None
    ) -> Label:
        label_repo = LabelRepository(str(project_id), self.db_session)
        try:
            db_label = label_repo.save(
                LabelDB(
                    id=str(label_id) if label_id is not None else None,
                    project_id=str(project_id),
                    name=name,
                    color=color if color else random_color(),
                    hotkey=hotkey,
                )
            )
            return Label.model_validate(db_label)
        except UniqueConstraintIntegrityError:
            raise DuplicateLabelsError
        except PrimaryKeyIntegrityError:
            raise ResourceWithIdAlreadyExistsError(ResourceType.LABEL, str(label_id))

    def list_all(self, project_id: UUID) -> list[Label]:
        label_repo = LabelRepository(str(project_id), self.db_session)
        db_labels = label_repo.list_all()
        return [Label.model_validate(db_label) for db_label in db_labels]

    def list_ids(self, project_id: UUID) -> list[UUID]:
        label_repo = LabelRepository(str(project_id), self.db_session)
        db_ids = label_repo.list_ids()
        return [UUID(db_id) for db_id in db_ids]

    def update_labels(
        self,
        project: Project,
        labels_to_add: list[Label],
        labels_to_remove: list[LabelReference],
        labels_to_edit: list[LabelUpdateInfo],
    ) -> list[Label]:
        """
        Update labels for a given project by adding, removing, and editing labels.

        Validates that the resulting number of labels satisfies project task constraints
        (e.g., multi-class classification requires at least two labels, and every project
        requires at least one label). Also validates that labels to remove or edit exist
        in the project before applying changes.

        Args:
            project (Project): The project whose labels to update.
            labels_to_add (list[Label]): Labels to be added to the project.
            labels_to_remove (list[LabelReference]): Labels to be removed from the project.
            labels_to_edit (list[LabelUpdateInfo]): Labels within the project to be edited.

        Returns:
            list[Label]: The full list of labels for the project after all updates have been applied.

        Raises:
            ValueError: If the resulting number of labels violates project task constraints.
            ResourceNotFoundError: If any labels to remove or edit do not exist in the project.
            DuplicateLabelsError: If any label names or hotkeys conflict with existing labels.
            ResourceWithIdAlreadyExistsError: If a label to add has an ID that already exists.
        """

        # Validate minimal number of labels satisfies project task constraints
        existing_ids = self.list_ids(project_id=project.id)
        new_number_of_labels = len(existing_ids) - len(labels_to_remove) + len(labels_to_add)
        if (
            project.task.task_type is TaskType.CLASSIFICATION
            and project.task.exclusive_labels
            and new_number_of_labels < 2
        ):
            raise ValueError(
                f"Multi-class classifications requires at least two labels, but after this label update the total "
                f"number of labels is {new_number_of_labels}."
            )
        if new_number_of_labels == 0:
            raise ValueError(
                f"A project requires at least one label, but after this label update the total number of labels is "
                f"{new_number_of_labels}."
            )

        # Validate labels to remove or edit exist in project
        if missing_ids_to_remove := [label.id for label in labels_to_remove if label.id not in existing_ids]:
            raise ResourceNotFoundError(
                resource_type=ResourceType.LABEL,
                resource_id=str(missing_ids_to_remove[0]),
                message="One or more labels to remove do not exist in the project",
            )
        if missing_ids_to_edit := [label.id for label in labels_to_edit if label.id not in existing_ids]:
            raise ResourceNotFoundError(
                resource_type=ResourceType.LABEL,
                resource_id=str(missing_ids_to_edit[0]),
                message="One or more labels to edit do not exist in the project",
            )

        for label_to_edit in labels_to_edit:
            self._update_label(
                project_id=project.id,
                label_id=label_to_edit.id,
                new_name=label_to_edit.new_name,
                new_color=label_to_edit.new_color,
                new_hotkey=label_to_edit.new_hotkey,
            )
        for label_to_remove in labels_to_remove:
            self._delete_label(project_id=project.id, label_id=label_to_remove.id)
        for label_to_add in labels_to_add:
            self.create_label(
                project_id=project.id,
                label_id=label_to_add.id,
                name=label_to_add.name,
                color=label_to_add.color,
                hotkey=label_to_add.hotkey,
            )
        return self.list_all(project_id=project.id)

    def _update_label(
        self, project_id: UUID, label_id: UUID, new_name: str | None, new_color: str | None, new_hotkey: str | None
    ) -> Label:
        label_repo = LabelRepository(str(project_id), self.db_session)
        try:
            db_label = label_repo.get_by_id(str(label_id))
            if db_label is None:
                raise ResourceNotFoundError(ResourceType.LABEL, str(label_id))
            db_label = label_repo.update(
                LabelDB(
                    id=db_label.id,
                    project_id=str(project_id),
                    name=new_name,
                    color=new_color,
                    hotkey=new_hotkey,
                )
            )
            return Label.model_validate(db_label)
        except UniqueConstraintIntegrityError:
            raise DuplicateLabelsError

    def _delete_label(self, project_id: UUID, label_id: UUID) -> None:
        label_repo = LabelRepository(str(project_id), self.db_session)
        if not label_repo.delete(str(label_id)):
            raise ResourceNotFoundError(ResourceType.LABEL, str(label_id))
