# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from app.db.schema import LabelDB, ProjectDB
from app.schemas.label import LabelCreate, LabelEdit, LabelRemove
from app.services import ResourceType, ResourceWithIdAlreadyExistsError
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
    ) -> None:
        """
        Test creating a label for the project.

        Verifies that:
        - A new label can be created for the existing project
        """
        db_project, _ = fxt_stored_project_with_labels
        label = LabelCreate(name="bird", color="#4500FF", hotkey="b")

        created_label = fxt_label_service.create_label(project_id=UUID(db_project.id), label=label)

        assert (
            created_label.name == label.name
            and created_label.color == label.color
            and created_label.hotkey == label.hotkey
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
            fxt_label_service.create_label(
                project_id=UUID(db_project.id), label=LabelCreate(name="cat", color="#4500FF", hotkey="b")
            )

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
            fxt_label_service.create_label(
                project_id=UUID(db_project.id), label=LabelCreate(name="bird", color="#4500FF", hotkey="c")
            )

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
                label=LabelCreate(id=UUID(existing_label_id), name="bird", color="#4500FF", hotkey="b"),
            )

        assert exc_info.value.resource_type == ResourceType.LABEL
        assert exc_info.value.resource_id == existing_label_id

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

    """
    Integration tests for LabelService's update_labels_in_project method.

    These tests verify the transactional behavior of adding, updating, and removing
    labels in a project, including the specific execution order and error handling.

    The tests focus on:
    - Successful operations with various combinations of add/update/remove
    - Uniqueness constraint violations (name, hotkey)
    - Transactional integrity (all-or-nothing behavior)
    - Execution order compliance (update → remove → add)
    """

    def test_add_labels_to_project(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
        db_session: Session,
    ) -> None:
        """
        Test adding new labels to a project without conflicts.

        Verifies that:
        - New labels are successfully added to the project
        - The returned list includes all labels (original + new)
        """
        db_project, _ = fxt_stored_project_with_labels

        new_labels = [
            LabelCreate(name="mouse", color="#0000FF", hotkey="r"),
            LabelCreate(name="bird", color="#FFFF00", hotkey="b"),
        ]
        labels = fxt_label_service.update_labels_in_project(UUID(db_project.id), new_labels, None, None)

        assert len(labels) == 4
        assert new_labels[0].name in [label.name for label in labels]
        assert new_labels[1].name in [label.name for label in labels]

    @pytest.mark.parametrize("new_label_attrs", [{"name": "cat"}, {"hotkey": "c"}])
    def test_add_existing_label_to_project(
        self,
        new_label_attrs,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
        db_session: Session,
    ) -> None:
        """
        Test that adding labels with duplicate attributes raises DuplicateLabelsError.

        Parametrized to test all uniqueness constraints:
        - Duplicate name
        - Duplicate hotkey

        Verifies proper error message and exception type.
        """
        db_project, _ = fxt_stored_project_with_labels

        new_label = LabelCreate(name="mouse", color="#0000FF", hotkey="r")
        new_label = new_label.model_copy(update=new_label_attrs)
        with pytest.raises(DuplicateLabelsError):
            fxt_label_service.update_labels_in_project(UUID(db_project.id), [new_label], None, None)

    def test_remove_labels_from_project(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
        db_session: Session,
    ) -> None:
        """
        Test removing all labels from a project.

        Verifies that:
        - Labels are successfully removed
        - Empty list is returned when all labels are removed
        - Operation completes without errors
        """
        db_project, db_labels = fxt_stored_project_with_labels

        labels = fxt_label_service.update_labels_in_project(
            UUID(db_project.id), None, None, [LabelRemove(id=UUID(label.id)) for label in db_labels]
        )

        assert len(labels) == 0

    def test_edit_labels_in_project(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
        db_session: Session,
    ) -> None:
        """
        Test updating existing labels' properties.

        Verifies that:
        - Label properties are successfully updated
        - The number of labels remains the same
        - Updated properties are reflected in the returned labels
        """
        db_project, db_labels = fxt_stored_project_with_labels

        labels_to_edit = [
            LabelEdit(
                id=UUID(db_label.id),
                new_name=f"edited_{db_label.name}",
                new_color="#FF0000",
                new_hotkey=None,
            )
            for db_label in db_labels
        ]
        labels = fxt_label_service.update_labels_in_project(UUID(db_project.id), None, labels_to_edit, None)

        assert len(labels) == 2
        assert {label.name for label in labels} == {labels_to_edit.new_name for labels_to_edit in labels_to_edit}
        assert {label.color for label in labels} == {labels_to_edit.new_color for labels_to_edit in labels_to_edit}

    def test_update_and_add_existing_label_in_project(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
        db_session: Session,
    ) -> None:
        """
        Test that updating and adding conflicting labels in same operation raises error.

        Verifies transactional behavior where a conflict in any part of the
        operation causes the entire transaction to fail and roll back.
        """
        db_project, db_labels = fxt_stored_project_with_labels

        with pytest.raises(DuplicateLabelsError):
            fxt_label_service.update_labels_in_project(
                UUID(db_project.id),
                [LabelCreate(name="mouse", color="#0000FF", hotkey="r")],
                [LabelEdit(id=UUID(db_labels[0].id), new_name="mouse", new_color="#FF0000", new_hotkey=None)],
                None,
            )

    def test_remove_update_add_combined_operation(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
        db_session: Session,
    ) -> None:
        """
        Test complex operation combining removal, update, and addition.

        Verifies the execution order (update → remove → add) works correctly
        and all operations are applied transactionally.
        """
        db_project, db_labels = fxt_stored_project_with_labels

        # Remove one label, update another, add a new one
        label_to_remove = LabelRemove(id=UUID(db_labels[0].id))
        label_to_update = LabelEdit(
            id=UUID(db_labels[1].id), new_name="updated_name", new_color="#FF0000", new_hotkey=None
        )
        new_label = LabelCreate(name="new_label", color="#123456", hotkey="x")

        labels = fxt_label_service.update_labels_in_project(
            UUID(db_project.id), [new_label], [label_to_update], [label_to_remove]
        )

        assert len(labels) == 2
        assert any(label.name == "updated_name" for label in labels)
        assert any(label.name == "new_label" for label in labels)
        assert not any(label.id == label_to_remove.id for label in labels)

    def test_remove_update_same_label_operation(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
        db_session: Session,
    ) -> None:
        """
        Test that updating and removing conflicting labels in same operation works.

        Verifies the execution order (update → remove) works correctly.
        """
        db_project, db_labels = fxt_stored_project_with_labels

        # Update the label and then remove it
        label_to_update = LabelEdit(
            id=UUID(db_labels[0].id), new_name="updated_name", new_color="#FF0000", new_hotkey=None
        )
        label_to_remove = LabelRemove(id=UUID(db_labels[0].id))

        labels = fxt_label_service.update_labels_in_project(
            UUID(db_project.id), None, [label_to_update], [label_to_remove]
        )

        assert len(labels) == 1
        assert labels[0].name != "updated_name"
        assert not any(label.id == label_to_remove.id for label in labels)

    def test_empty_operations(
        self,
        fxt_stored_project_with_labels: tuple[ProjectDB, list[LabelDB]],
        fxt_label_service: LabelService,
        db_session: Session,
    ) -> None:
        """
        Test that calling with all None parameters returns current labels unchanged.

        Verifies graceful handling of no-op calls.
        """
        db_project, db_labels = fxt_stored_project_with_labels

        original_labels = db_labels.copy()

        labels = fxt_label_service.update_labels_in_project(UUID(db_project.id), None, None, None)

        assert len(labels) == len(original_labels)
        assert {str(label.id) for label in labels} == {label.id for label in original_labels}
