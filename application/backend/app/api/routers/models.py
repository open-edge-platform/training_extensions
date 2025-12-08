# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import zipfile
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_model_service, get_project
from app.api.schemas import ModelView, ProjectView
from app.api.validators import ModelID
from app.services import ModelService, ResourceInUseError, ResourceNotFoundError

router = APIRouter(prefix="/api/projects/{project_id}/models", tags=["Models"])


@router.get(
    "",
    response_model=list[ModelView],
    responses={
        status.HTTP_200_OK: {"description": "List of available models"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
def list_models(
    project: Annotated[ProjectView, Depends(get_project)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> list[ModelView]:
    """Get all models in a project."""
    try:
        return [ModelView.model_validate(obj, from_attributes=True) for obj in model_service.list_models(project.id)]
    except ResourceNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")


@router.get(
    "/{model_id}",
    response_model=ModelView,
    responses={
        status.HTTP_200_OK: {"description": "Model found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project or model ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project or model not found"},
    },
)
def get_model(
    project: Annotated[ProjectView, Depends(get_project)],
    model_id: ModelID,
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> ModelView:
    """Get a specific model by ID."""
    try:
        model_revision = model_service.get_model(project_id=project.id, model_id=model_id)
        return ModelView.model_validate(model_revision, from_attributes=True)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get(
    "/{model_id}/binary",
    responses={
        status.HTTP_200_OK: {"description": "Model weights in OpenVINO format (zip archive)"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project or model ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project or model not found"},
    },
)
def download_model_binary(
    project: Annotated[ProjectView, Depends(get_project)],
    model_id: ModelID,
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> StreamingResponse:
    """Download trained model weights in OpenVINO format as a zip archive containing model.xml and model.bin files."""
    try:
        # Verify the model exists and get the model directory
        model_dir = model_service.get_model_files_path(project_id=project.id, model_id=model_id)

        # Create an in-memory zip file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            xml_file = model_dir / "model.xml"
            bin_file = model_dir / "model.bin"

            zip_file.write(xml_file, arcname="model.xml")
            zip_file.write(bin_file, arcname="model.bin")

        zip_buffer.seek(0)

        # Assume FP16 precision by default
        filename = f"model-{model_id}-fp16.zip"

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete(
    "/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {
            "description": "Model configuration successfully deleted",
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project or model ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project or model not found"},
        status.HTTP_409_CONFLICT: {"description": "Model is used by at least one pipeline"},
    },
)
def delete_model(
    project: Annotated[ProjectView, Depends(get_project)],
    model_id: ModelID,
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> None:
    """Delete a model from a project."""
    try:
        model_service.delete_model(project_id=project.id, model_id=model_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ResourceInUseError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
