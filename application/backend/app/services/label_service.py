# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from sqlalchemy.orm import Session

from app.db.schema import LabelDB
from app.models import Label
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

    def update_label(
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

    def delete_label(self, project_id: UUID, label_id: UUID) -> None:
        label_repo = LabelRepository(str(project_id), self.db_session)
        if not label_repo.delete(str(label_id)):
            raise ResourceNotFoundError(ResourceType.LABEL, str(label_id))
