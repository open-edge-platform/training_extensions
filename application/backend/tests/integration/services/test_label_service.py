# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from app.db.schema import LabelDB, ProjectDB
from app.services import ResourceNotFoundError, ResourceType, ResourceWithIdAlreadyExistsError
from app.services.label_service import DuplicateLabelsError, LabelService


@pytest.fixture
def fxt_label_service(db_session: Session) -> LabelService:
    return LabelService(db_session)


@pytest.fixture
def fxt_stored_project_with_labels(
    fxt_db_projects: list[ProjectDB],
    fxt_db_labels: list[LabelDB],
    db_session: Session,
) -> tuple[ProjectDB, list[LabelDB]]:
    db_project = fxt_db_projects[0]
    db_session.add(db_project)
    db_session.flush()

    for label in fxt_db_labels:
        label.project_id = db_project.id
    db_session.add_all(fxt_db_labels)
    db_session.flush()

    return db_project, fxt_db_labels


class TestLabelServiceIntegration:
    def test_create_label(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
        db_session: Session,
    ) -> None:
        """
        Test creating a label for the project.

        Verifies that:
        - A new label can be created for the existing project
        """
        db_project, _ = fxt_stored_project_with_labels
        created_label = fxt_label_service.create_label(
            project_id=UUID(db_project.id), name="bird", color="#4500FF", hotkey="b"
        )

        assert created_label.name == "bird" and created_label.color == "#4500FF" and created_label.hotkey == "b"

        created_db_label = db_session.query(LabelDB).filter(LabelDB.id == str(created_label.id)).one()
        assert (
            created_db_label.name == "bird" and created_db_label.color == "#4500FF" and created_db_label.hotkey == "b"
        )

    def test_create_label_duplicate_name(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test creating a label with duplicating name for the project.

        Verifies that:
        - A new label with duplicating name cannot be created
        """
        db_project, _ = fxt_stored_project_with_labels

        with pytest.raises(DuplicateLabelsError):
            fxt_label_service.create_label(project_id=UUID(db_project.id), name="cat", color="#4500FF", hotkey="b")

    def test_create_label_duplicate_hotkey(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test creating a label with duplicating hotkey for the project.

        Verifies that:
        - A new label with duplicating hotkey cannot be created
        """
        db_project, _ = fxt_stored_project_with_labels

        with pytest.raises(DuplicateLabelsError):
            fxt_label_service.create_label(project_id=UUID(db_project.id), name="bird", color="#4500FF", hotkey="c")

    def test_create_label_duplicate_id(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test creating a label with duplicating ID for the project.

        Verifies that:
        - A new label with duplicating ID cannot be created
        """
        db_project, db_labels = fxt_stored_project_with_labels
        existing_label_id = db_labels[0].id

        with pytest.raises(ResourceWithIdAlreadyExistsError) as exc_info:
            fxt_label_service.create_label(
                project_id=UUID(db_project.id),
                label_id=UUID(existing_label_id),
                name="bird",
                color="#4500FF",
                hotkey="b",
            )

        assert exc_info.value.resource_type == ResourceType.LABEL
        assert exc_info.value.resource_id == existing_label_id

    def test_update_label(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
        db_session: Session,
    ) -> None:
        """
        Test updating a label for the project.

        Verifies that:
        - An existing label can be updated for the existing project
        """
        db_project, db_labels = fxt_stored_project_with_labels
        updated_label = fxt_label_service.update_label(
            project_id=UUID(db_project.id),
            label_id=UUID(db_labels[0].id),
            new_name="bird",
            new_color="#4500FF",
            new_hotkey="b",
        )

        assert updated_label.name == "bird" and updated_label.color == "#4500FF" and updated_label.hotkey == "b"

        updated_db_label = db_session.query(LabelDB).filter(LabelDB.id == db_labels[0].id).one()
        assert (
            updated_db_label.name == "bird" and updated_db_label.color == "#4500FF" and updated_db_label.hotkey == "b"
        )

    def test_update_label_duplicate_name(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
        db_session: Session,
    ) -> None:
        """
        Test updating a label to have a duplicating name for the project.

        Verifies that:
        - A label cannot be updated to have a duplicating name
        """
        db_project, db_labels = fxt_stored_project_with_labels
        with pytest.raises(DuplicateLabelsError):
            fxt_label_service.update_label(
                project_id=UUID(db_project.id),
                label_id=UUID(db_labels[0].id),
                new_name=db_labels[1].name,
                new_color="#4500FF",
                new_hotkey="b",
            )

    def test_update_label_duplicate_hotkey(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
        db_session: Session,
    ) -> None:
        """
        Test updating a label to have a duplicating hotkey for the project.

        Verifies that:
        - A label cannot be updated to have a duplicating hotkey
        """
        db_project, db_labels = fxt_stored_project_with_labels
        with pytest.raises(DuplicateLabelsError):
            fxt_label_service.update_label(
                project_id=UUID(db_project.id),
                label_id=UUID(db_labels[0].id),
                new_name="bird",
                new_color="#4500FF",
                new_hotkey=db_labels[1].hotkey,
            )

    def test_delete_label(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
        db_session: Session,
    ) -> None:
        """
        Test deleting a label for the project.

        Verifies that:
        - An existing label can be deleted for the existing project
        """
        db_project, db_labels = fxt_stored_project_with_labels
        fxt_label_service.delete_label(project_id=UUID(db_project.id), label_id=UUID(db_labels[0].id))

        assert db_session.query(LabelDB).filter(LabelDB.id == db_labels[0].id).one_or_none() is None

    def test_delete_non_existing_label(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
        db_session: Session,
    ) -> None:
        """
        Test deleting a label for the project.

        Verifies that:
        - An existing label can be deleted for the existing project
        """
        non_existing_label_id = uuid4()
        db_project, _ = fxt_stored_project_with_labels
        with pytest.raises(ResourceNotFoundError) as exc_info:
            fxt_label_service.delete_label(project_id=UUID(db_project.id), label_id=non_existing_label_id)

        assert exc_info.value.resource_type == ResourceType.LABEL
        assert exc_info.value.resource_id == str(non_existing_label_id)

    def test_list_all(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test getting a list of labels for the project.

        Verifies that:
        - Stored project labels can be obtained successfully
        """
        db_project, db_labels = fxt_stored_project_with_labels

        labels = fxt_label_service.list_all(UUID(db_project.id))

        assert len(labels) == len(db_labels)
        assert [label.name for label in labels] == [db_label.name for db_label in db_labels]

    def test_list_all_non_existing_project(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test getting a list of labels for non-existing project.

        Verifies that:
        - In case of non-existing project empty labels list is returned
        """
        db_project, _ = fxt_stored_project_with_labels

        labels = fxt_label_service.list_all(uuid4())

        assert len(labels) == 0

    def test_list_ids(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test getting a list of labels ID's for the project.

        Verifies that:
        - Stored project labels ID's can be obtained successfully
        """
        db_project, db_labels = fxt_stored_project_with_labels

        ids = fxt_label_service.list_ids(UUID(db_project.id))

        assert len(ids) == len(db_labels)
        assert ids == [UUID(db_label.id) for db_label in db_labels]

    def test_list_ids_non_existing_project(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test getting a list of labels ID's for non-existing project.

        Verifies that:
        - In case of non-existing project empty labels ID's list is returned
        """
        db_project, _ = fxt_stored_project_with_labels

        ids = fxt_label_service.list_ids(uuid4())

        assert len(ids) == 0
