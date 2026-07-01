# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import cast
from uuid import UUID

from behave import given, then, when
from behave.runner import Context
from requests import Session

from app.api.schemas import ProjectView
from app.api.schemas.jobs.dataset_export import ExportDatasetMetadata
from app.models import DatasetFormat
from tests.bdd.utils import export_dataset, prepare_dataset


@given("the project dataset is exported in {export_format} format")  # pyrefly: ignore
def step_project_is_exported(context: Context, export_format: str) -> None:
    """Add multiple random unannotated images to the dataset."""
    project = cast(ProjectView, context.project)
    export_format = DatasetFormat(export_format.lower())
    job = export_dataset(
        session=cast(Session, context.session),
        base_url=str(context.base_url),
        project_id=str(project.id),
        export_format=export_format,
    )
    context.dataset_id = cast(ExportDatasetMetadata, job.metadata).dataset_id
    context.export_format = export_format


@when("I prepare the staged dataset archive for import")  # pyrefly: ignore
def step_prepare_dataset_for_import(context: Context) -> None:
    dataset_id = cast(UUID, context.dataset_id)
    prepare_dataset(
        session=cast(Session, context.session), base_url=str(context.base_url), staged_dataset_id=str(dataset_id)
    )


@then("the staged dataset is ready for import")  # pyrefly: ignore
def step_dataset_archive_ready(context: Context) -> None:
    session = cast(Session, context.session)
    response = session.get(f"{str(context.base_url)}/api/staged_datasets/{context.dataset_id}")
    assert response.status_code == 200, (
        f"Expected status code 200, got {response.status_code}, response: {response.text}"
    )
    dataset_info = response.json()
    assert dataset_info, "Expected dataset info in response, got None"
    dataset_format = dataset_info["format"]
    assert dataset_format == DatasetFormat.GETI, f"Expected dataset format '{DatasetFormat.GETI}', got {dataset_format}"
    assert dataset_info["ready_for_import"], "Expected dataset to be ready for import, but it is not"
    assert dataset_info["metadata"], "Expected dataset metadata in response, got None"
