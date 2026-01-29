# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.params import Depends
from starlette.responses import StreamingResponse

from app.api.dependencies import get_staged_dataset_service
from app.api.schemas import StagedDatasetView
from app.api.validators import StagedDatasetID
from app.services.staged_dataset_service import StagedDatasetService

router = APIRouter(prefix="/api/staged_datasets", tags=["Dataset Import/Export"])


@router.post(
    "",
    response_model=StagedDatasetView,
    responses={
        status.HTTP_201_CREATED: {"description": "Dataset archive uploaded successfully"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Invalid archive has been uploaded"},
    },
)
async def upload_dataset_archive(
    file: Annotated[UploadFile, File()],
    staged_datasets_service: Annotated[StagedDatasetService, Depends(get_staged_dataset_service)],
) -> StagedDatasetView:
    """Upload dataset archive to the staging area"""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded archive must have a filename.",
        )

    try:
        staged_dataset = await staged_datasets_service.upload(
            filename=file.filename,
            chunk_reader=lambda: file.read(1024 * 1024),
        )
        return StagedDatasetView.model_validate(staged_dataset, from_attributes=True)
    finally:
        await file.close()


@router.get(
    "",
    response_model=list[StagedDatasetView],
    responses={
        status.HTTP_200_OK: {
            "description": "List of staged dataset archives",
        },
    },
)
async def list_datasets(
    staged_dataset_service: Annotated[StagedDatasetService, Depends(get_staged_dataset_service)],
) -> list[StagedDatasetView]:
    """List all datasets from the staging area"""
    return [StagedDatasetView.model_validate(item, from_attributes=True) for item in staged_dataset_service.list_all()]


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
