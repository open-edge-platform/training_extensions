from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.openapi.models import Example

from app.api.dependencies import get_model_id
from app.schemas import Model
from app.services import ModelAlreadyExistsError, ModelService, ResourceInUseError, ResourceNotFoundError

router = APIRouter(prefix="/api/models", tags=["Models"])


UPDATE_MODEL_BODY_EXAMPLES = {
    "rename_model": Example(
        summary="Rename model",
        description="Change the name of the model",
        value={
            "name": "New Model Name",
        },
    )
}


# TODO update this endpoint
@router.post("", status_code=status.HTTP_201_CREATED)
async def add_model(
    model_name: Annotated[str, Query(description="Name for the model files")],
    xml_file: Annotated[UploadFile, File()],
    bin_file: Annotated[UploadFile, File()],
) -> Model:
    """
    Upload a new model

    NOTE: this endpoint will be replaced by preconfigured model selection
    """
    # Validate file extensions
    if not xml_file.filename or not xml_file.filename.endswith(".xml"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="The model XML file must have .xml extension"
        )
    if not bin_file.filename or not bin_file.filename.endswith(".bin"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="The model BIN file must have .bin extension"
        )

    try:
        return await ModelService().add_model(model_name, xml_file, bin_file)
    except ModelAlreadyExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.get(
    "",
    responses={
        status.HTTP_200_OK: {"description": "List of available models", "model": list[Model]},
    },
)
async def list_models() -> list[Model]:
    """Get information about available models"""
    return ModelService().list_models()


@router.get(
    "/{model_id}",
    responses={
        status.HTTP_200_OK: {"description": "Model found", "model": Model},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid model ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Model not found"},
    },
)
async def get_model(model_id: Annotated[UUID, Depends(get_model_id)]) -> Model:
    """Get information about a specific model"""
    try:
        return ModelService().get_model_by_id(model_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.patch(
    "/{model_id}",
    responses={
        status.HTTP_200_OK: {"description": "Model successfully updated", "model": Model},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid model ID or request body"},
        status.HTTP_404_NOT_FOUND: {"description": "Model not found"},
    },
)
async def update_model_metadata(
    model_id: Annotated[UUID, Depends(get_model_id)],
    model_metadata: Annotated[dict, Body(openapi_examples=UPDATE_MODEL_BODY_EXAMPLES)],
) -> Model:
    """Update the metadata of an existing model"""
    if "format" in model_metadata:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The 'format' field cannot be changed")
    try:
        return ModelService().update_model(model_id, model_metadata)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete(
    "/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {
            "description": "Model configuration successfully deleted",
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid model ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Model not found"},
        status.HTTP_409_CONFLICT: {"description": "Model is used by at least one pipeline"},
    },
)
async def delete_model(model_id: Annotated[UUID, Depends(get_model_id)]) -> None:
    """Delete a model"""
    try:
        ModelService().delete_model_by_id(model_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ResourceInUseError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


# TODO remove this endpoint
@router.post("/{model_name}:activate", deprecated=True)
async def activate_model(model_name: str) -> Model:
    """
    Activate a model

    NOTE: this endpoint will be removed; use instead `PATCH /api/pipelines/{pipeline_id}` to change the active model
    """
    try:
        ModelService().activate_model(model_name)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    return Model(name=model_name)
