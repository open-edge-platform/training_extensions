# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import os
import zipfile
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from fastapi.openapi.models import Example
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_model_service, get_project
from app.api.schemas import ModelView, ProjectView
from app.api.validators import ModelID
from app.models.model_revision import ModelFormat
from app.services import ModelService, ResourceInUseError, ResourceNotFoundError, ResourceType

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
        model_views = []
        for model_revision in model_service.list_models(project.id):
            model_variants = model_service.get_model_variants(project_id=project.id, model_id=model_revision.id)
            model_views.append(model_revision.model_dump() | {"variants": model_variants})
        return [ModelView.model_validate(model_view, from_attributes=True) for model_view in model_views]
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
        model_variants = model_service.get_model_variants(project_id=project.id, model_id=model_id)
        model_view = model_revision.model_dump() | {"variants": model_variants}
        return ModelView.model_validate(model_view, from_attributes=True)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get(
    "/{model_id}/binary",
    responses={
        status.HTTP_200_OK: {"description": "Model weights in either OpenVINO or ONNX format (zip archive)"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project or model ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project or model not found"},
    },
)
def download_model_binary(
    project: Annotated[ProjectView, Depends(get_project)],
    model_id: ModelID,
    model_service: Annotated[ModelService, Depends(get_model_service)],
    format: Annotated[ModelFormat, Query()] = ModelFormat.OPENVINO,
) -> StreamingResponse:
    """Download trained model weights in a desired format as a zip archive"""
    try:
        files_exist, paths = model_service.get_model_binary_files(
            project_id=project.id, model_id=model_id, format=format
        )
        if not files_exist:
            raise ResourceNotFoundError(
                resource_type=ResourceType.MODEL,
                resource_id=f"{model_id} with format {format.value}",
            )

        # Create an in-memory zip file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            for path in paths:
                zip_file.write(path, arcname=os.path.split(path)[1])

        zip_buffer.seek(0)

        precision = "fp16" if format != ModelFormat.PYTORCH else "fp32"
        filename = f"model-{model_id}-{format.value}-{precision}.zip"

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


UPDATE_MODEL_BODY_DESCRIPTION = """
Update name of model revision.
"""
UPDATE_MODEL_BODY_EXAMPLES = {
    "name": Example(
        summary="Update model name",
        description="Change the name of the model",
        value={
            "name": "new_model_name",
        },
    ),
}


@router.patch(
    "/{model_id}",
    response_model=ModelView,
    responses={
        status.HTTP_200_OK: {"description": "Model successfully renamed"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project or model ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Project or model not found"},
    },
)
def rename_model(
    project: Annotated[ProjectView, Depends(get_project)],
    model_id: ModelID,
    model_metadata: Annotated[
        dict,
        Body(
            description=UPDATE_MODEL_BODY_DESCRIPTION,
            openapi_examples=UPDATE_MODEL_BODY_EXAMPLES,
        ),
    ],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> ModelView:
    """Rename a model"""
    try:
        model_revision = model_service.rename_model(
            project_id=project.id, model_id=model_id, model_metadata=model_metadata
        )
        return ModelView.model_validate(model_revision, from_attributes=True)
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
    files_only: Annotated[bool, Query()] = False,
) -> None:
    """Delete a model from a project."""
    try:
        if files_only:
            model_service.delete_model_files(project_id=project.id, model_id=model_id)
        else:
            model_service.delete_model(project_id=project.id, model_id=model_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ResourceInUseError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
