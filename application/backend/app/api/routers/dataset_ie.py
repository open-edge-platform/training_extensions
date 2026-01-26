# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated

from fastapi import APIRouter, File, UploadFile, status
from starlette.responses import StreamingResponse

from app.api.schemas import StagedDatasetView
from app.api.validators import StagedDatasetID

router = APIRouter(prefix="/api/staged_datasets", tags=["Dataset Import/Export"])


@router.post(
    "",
    response_model=StagedDatasetView,
    responses={
        status.HTTP_200_OK: {"description": "Dataset archive uploaded successfully"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Invalid archive has been uploaded"},
    },
)
def upload_dataset_archive(file: Annotated[UploadFile, File()]) -> StagedDatasetView:
    """Upload dataset archive to the staging area"""
    raise NotImplementedError


@router.get(
    "",
    response_model=list[StagedDatasetView],
    responses={
        status.HTTP_200_OK: {
            "description": "List of staged dataset archives",
        },
    },
)
def list_datasets() -> list[StagedDatasetView]:
    """List all datasets from the staging area"""
    raise NotImplementedError


@router.get(
    "/{staged_dataset_id}",
    response_model=StagedDatasetView,
    responses={
        status.HTTP_200_OK: {"description": "Staged dataset found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid staged dataset ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Staged dataset not found"},
    },
)
def get_dataset(staged_dataset_id: StagedDatasetID) -> StagedDatasetView:
    """Get info about the staged dataset from the staging area"""
    raise NotImplementedError


@router.get(
    "/{staged_dataset_id}/zip",
    responses={
        status.HTTP_200_OK: {"description": "Staged dataset found"},
        status.HTTP_404_NOT_FOUND: {"description": "Staged dataset not found"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "Staged dataset is not in zip format ready for download"
        },
    },
)
def download_archive(staged_dataset_id: StagedDatasetID) -> StreamingResponse:
    """Download the staged dataset archive from the staging area"""
    raise NotImplementedError


@router.delete(
    "/{staged_dataset_id}",
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Staged dataset successfully deleted"},
        status.HTTP_404_NOT_FOUND: {"description": "Staged dataset not found"},
    },
)
def delete_dataset(staged_dataset_id: StagedDatasetID) -> None:
    """Delete the staged dataset from the staging area"""
    raise NotImplementedError
