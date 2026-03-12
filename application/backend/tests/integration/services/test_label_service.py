# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from app.db.schema import LabelDB, ProjectDB
from app.models import Label
from app.models.label import LabelReference, LabelUpdateInfo
from app.models.project import Project
from app.models.task import Task, TaskType
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


def _db_project_to_project(db_project: ProjectDB) -> Project:
    return Project(
        id=UUID(db_project.id),
        name=db_project.name,
        task=Task(
            task_type=TaskType(db_project.task_type),
            labels=[],
        ),
    )


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

    def test_update_labels_add_edit_delete(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
        db_session: Session,
    ) -> None:
        """
        Test combined add, edit, and delete via update_labels.

        Verifies that:
        - All three operations are applied in a single call
        """
        db_project, db_labels = fxt_stored_project_with_labels
        project = _db_project_to_project(db_project)
        edit_id = UUID(db_labels[0].id)
        remove_id = UUID(db_labels[1].id)
        new_label_id = uuid4()

        result = fxt_label_service.update_labels(
            project=project,
            labels_to_add=[Label(id=new_label_id, project_id=project.id, name="bird", color="#0000FF", hotkey="b")],
            labels_to_remove=[LabelReference(id=remove_id)],
            labels_to_edit=[LabelUpdateInfo(id=edit_id, new_name="updated_cat", new_color="#121212", new_hotkey=None)],
        )

        result_ids = {lbl.id for lbl in result}
        assert edit_id in result_ids
        assert remove_id not in result_ids
        assert new_label_id in result_ids

        edited = next(lbl for lbl in result if lbl.id == edit_id)
        assert edited.name == "updated_cat" and edited.color == "#121212"

        added = next(lbl for lbl in result if lbl.id == new_label_id)
        assert added.name == "bird" and added.color == "#0000FF" and added.hotkey == "b"

        assert db_session.query(LabelDB).filter(LabelDB.id == str(remove_id)).one_or_none() is None

    def test_update_labels_edit_duplicate_name(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test editing a label to have a duplicate name raises DuplicateLabelsError.
        """
        db_project, db_labels = fxt_stored_project_with_labels
        project = _db_project_to_project(db_project)

        with pytest.raises(DuplicateLabelsError):
            fxt_label_service.update_labels(
                project=project,
                labels_to_add=[],
                labels_to_remove=[],
                labels_to_edit=[
                    LabelUpdateInfo(
                        id=UUID(db_labels[0].id),
                        new_name=db_labels[1].name,
                        new_color="#4500FF",
                        new_hotkey="b",
                    )
                ],
            )

    def test_update_labels_edit_duplicate_hotkey(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test editing a label to have a duplicate hotkey raises DuplicateLabelsError.
        """
        db_project, db_labels = fxt_stored_project_with_labels
        project = _db_project_to_project(db_project)

        with pytest.raises(DuplicateLabelsError):
            fxt_label_service.update_labels(
                project=project,
                labels_to_add=[],
                labels_to_remove=[],
                labels_to_edit=[
                    LabelUpdateInfo(
                        id=UUID(db_labels[0].id),
                        new_name="bird",
                        new_color="#4500FF",
                        new_hotkey=db_labels[1].hotkey,
                    )
                ],
            )

    def test_update_labels_edit_nonexistent(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test editing a non-existent label raises ResourceNotFoundError.
        """
        db_project, _ = fxt_stored_project_with_labels
        project = _db_project_to_project(db_project)
        nonexistent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as exc_info:
            fxt_label_service.update_labels(
                project=project,
                labels_to_add=[],
                labels_to_remove=[],
                labels_to_edit=[
                    LabelUpdateInfo(id=nonexistent_id, new_name="bird", new_color="#4500FF", new_hotkey="b")
                ],
            )

        assert exc_info.value.resource_type == ResourceType.LABEL
        assert exc_info.value.resource_id == str(nonexistent_id)

    def test_update_labels_delete_nonexistent(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test deleting a non-existent label raises ResourceNotFoundError.
        """
        db_project, _ = fxt_stored_project_with_labels
        project = _db_project_to_project(db_project)
        nonexistent_id = uuid4()

        with pytest.raises(ResourceNotFoundError) as exc_info:
            fxt_label_service.update_labels(
                project=project,
                labels_to_add=[],
                labels_to_remove=[LabelReference(id=nonexistent_id)],
                labels_to_edit=[],
            )

        assert exc_info.value.resource_type == ResourceType.LABEL
        assert exc_info.value.resource_id == str(nonexistent_id)

    def test_update_labels_remove_all_raises_value_error(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test removing all labels raises ValueError (project requires at least one label).
        """
        db_project, db_labels = fxt_stored_project_with_labels
        project = _db_project_to_project(db_project)

        with pytest.raises(ValueError, match="A project requires at least one label"):
            fxt_label_service.update_labels(
                project=project,
                labels_to_add=[],
                labels_to_remove=[LabelReference(id=UUID(lbl.id)) for lbl in db_labels],
                labels_to_edit=[],
            )

    def test_update_labels_empty(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
    ) -> None:
        """
        Test update_labels with no changes returns all existing labels unchanged.
        """
        db_project, db_labels = fxt_stored_project_with_labels
        project = _db_project_to_project(db_project)

        result = fxt_label_service.update_labels(
            project=project,
            labels_to_add=[],
            labels_to_remove=[],
            labels_to_edit=[],
        )

        assert len(result) == len(db_labels)
        assert {lbl.id for lbl in result} == {UUID(lbl.id) for lbl in db_labels}

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
        fxt_stored_project_with_labels  # ensure fixture runs

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
        fxt_stored_project_with_labels  # ensure fixture runs

        ids = fxt_label_service.list_ids(uuid4())

        assert len(ids) == 0
