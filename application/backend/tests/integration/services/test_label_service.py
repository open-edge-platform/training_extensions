# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

import pytest
from sqlalchemy.orm import Session

from app.db.schema import ProjectDB
from app.schemas.label import Label
from app.services import ResourceAlreadyExistsError
from app.services.label_service import LabelService


@pytest.fixture
def fxt_label_service(db_session: Session) -> LabelService:
    return LabelService(db_session)


class TestLabelServiceIntegration:
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
        self, fxt_db_projects: list[ProjectDB], fxt_label_service: LabelService, db_session: Session
    ):
        """
        Test adding new labels to a project without conflicts.

        Verifies that:
        - New labels are successfully added to the project
        - The returned list includes all labels (original + new)
        """
        db_session.add(fxt_db_projects[0])
        db_session.flush()

        new_labels = [
            Label(name="mouse", color="#0000FF", hotkey="r"),
            Label(name="bird", color="#FFFF00", hotkey="b"),
        ]
        labels = fxt_label_service.update_labels_in_project(UUID(fxt_db_projects[0].id), new_labels, None, None)

        assert len(labels) == 4
        assert new_labels[0] in labels
        assert new_labels[1] in labels

    @pytest.mark.parametrize("new_label_attrs", [{"name": "cat"}, {"hotkey": "c"}])
    def test_add_existing_label_to_project(
        self, new_label_attrs, fxt_db_projects: list[ProjectDB], fxt_label_service: LabelService, db_session: Session
    ):
        """
        Test that adding labels with duplicate attributes raises ResourceAlreadyExistsError.

        Parametrized to test all uniqueness constraints:
        - Duplicate name
        - Duplicate hotkey

        Verifies proper error message and exception type.
        """
        db_session.add(fxt_db_projects[0])
        db_session.flush()

        new_label = Label(name="mouse", color="#0000FF", hotkey="r")
        new_label = new_label.model_copy(update=new_label_attrs)
        with pytest.raises(ResourceAlreadyExistsError) as exc_info:
            fxt_label_service.update_labels_in_project(UUID(fxt_db_projects[0].id), [new_label], None, None)

        assert str(exc_info.value) == "Label with the same name or hotkey already exists in this project."

    def test_remove_labels_from_project(
        self, fxt_db_projects: list[ProjectDB], fxt_label_service: LabelService, db_session: Session
    ):
        """
        Test removing all labels from a project.

        Verifies that:
        - Labels are successfully removed
        - Empty list is returned when all labels are removed
        - Operation completes without errors
        """
        db_project = fxt_db_projects[0]
        db_session.add(db_project)
        db_session.flush()

        labels = fxt_label_service.update_labels_in_project(
            UUID(db_project.id), None, None, [label.id for label in db_project.labels]
        )

        assert len(labels) == 0

    def test_edit_labels_in_project(
        self, fxt_db_projects: list[ProjectDB], fxt_label_service: LabelService, db_session: Session
    ):
        """
        Test updating existing labels' properties.

        Verifies that:
        - Label properties are successfully updated
        - The number of labels remains the same
        - Updated properties are reflected in the returned labels
        """
        db_project = fxt_db_projects[0]
        db_session.add(db_project)
        db_session.flush()

        labels_to_edit = [
            Label(
                id=db_label.id,
                name=f"edited_{db_label.name}",
            )  # type: ignore[call-arg]
            for db_label in db_project.labels
        ]
        labels = fxt_label_service.update_labels_in_project(UUID(db_project.id), None, labels_to_edit, None)

        assert len(labels) == 2
        assert {label.name for label in labels} == {labels_to_edit.name for labels_to_edit in labels_to_edit}

    def test_update_and_add_existing_label_in_project(
        self, fxt_db_projects: list[ProjectDB], fxt_label_service: LabelService, db_session: Session
    ):
        """
        Test that updating and adding conflicting labels in same operation raises error.

        Verifies transactional behavior where a conflict in any part of the
        operation causes the entire transaction to fail and roll back.
        """
        db_project = fxt_db_projects[0]
        db_session.add(db_project)
        db_session.flush()

        with pytest.raises(ResourceAlreadyExistsError) as exc_info:
            fxt_label_service.update_labels_in_project(
                UUID(db_project.id),
                [Label(name="mouse", color="#0000FF", hotkey="r")],
                [Label(id=db_project.labels[0].id, name="mouse")],  # type: ignore[call-arg]
                None,
            )

        assert str(exc_info.value) == "Label with the same name or hotkey already exists in this project."

    def test_remove_update_add_combined_operation(
        self, fxt_db_projects: list[ProjectDB], fxt_label_service: LabelService, db_session: Session
    ):
        """
        Test complex operation combining removal, update, and addition.

        Verifies the execution order (update → remove → add) works correctly
        and all operations are applied transactionally.
        """
        db_project = fxt_db_projects[0]
        db_session.add(db_project)
        db_session.flush()

        # Remove one label, update another, add a new one
        label_to_remove_id = db_project.labels[0].id
        label_to_update = Label(id=db_project.labels[1].id, name="updated_name")  # type: ignore[call-arg]
        new_label = Label(name="new_label", color="#123456", hotkey="x")

        labels = fxt_label_service.update_labels_in_project(
            UUID(db_project.id), [new_label], [label_to_update], [label_to_remove_id]
        )

        assert len(labels) == 2
        assert any(label.name == "updated_name" for label in labels)
        assert any(label.name == "new_label" for label in labels)
        assert not any(label.id == label_to_remove_id for label in labels)

    def test_remove_update_same_label_operation(
        self, fxt_db_projects: list[ProjectDB], fxt_label_service: LabelService, db_session: Session
    ):
        """
        Test that updating and removing conflicting labels in same operation works.

        Verifies the execution order (update → remove) works correctly.
        """
        db_project = fxt_db_projects[0]
        db_session.add(db_project)
        db_session.flush()

        # Update the label and then remove it
        label_to_update = Label(id=db_project.labels[0].id, name="updated_name")  # type: ignore[call-arg]
        label_to_remove_id = db_project.labels[0].id

        labels = fxt_label_service.update_labels_in_project(
            UUID(db_project.id), None, [label_to_update], [label_to_remove_id]
        )

        assert len(labels) == 1
        assert labels[0].name != "updated_name"
        assert not any(label.id == label_to_remove_id for label in labels)

    def test_empty_operations(
        self, fxt_db_projects: list[ProjectDB], fxt_label_service: LabelService, db_session: Session
    ):
        """
        Test that calling with all None parameters returns current labels unchanged.

        Verifies graceful handling of no-op calls.
        """
        db_project = fxt_db_projects[0]
        db_session.add(db_project)
        db_session.flush()

        original_labels = db_project.labels.copy()

        labels = fxt_label_service.update_labels_in_project(UUID(db_project.id), None, None, None)

        assert len(labels) == len(original_labels)
        assert {str(label.id) for label in labels} == {label.id for label in original_labels}
