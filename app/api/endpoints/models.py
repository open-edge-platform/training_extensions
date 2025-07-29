from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel

from app.api.dependencies import get_model_id
from app.schemas.model import Model
from app.services.model_service import ModelAlreadyExistsError, ModelNotFoundError, ModelService

router = APIRouter(prefix="/api/models", tags=["Models"])


UPDATE_MODEL_BODY_EXAMPLES = {
    "rename_model": {
        "summary": "Rename model",
        "description": "Change the name of the model",
        "value": {
            "name": "New Model Name",
        },
    },
}


class ModelResponse(BaseModel):
    model_name: str


class ModelsInfoResponse(BaseModel):
    active_model: str | None
    available_models: list[str]


@router.post("", status_code=status.HTTP_201_CREATED)
async def add_model(
    model_name: Annotated[str, Query(description="Name for the model files")],
    xml_file: Annotated[UploadFile, File()],
    bin_file: Annotated[UploadFile, File()],
) -> ModelResponse:  # TODO return schemas.model.Model
    """Upload a new model"""
    # Validate file extensions
    if not xml_file.filename.endswith(".xml"):
        raise HTTPException(status_code=400, detail="The model XML file must have .xml extension")
    if not bin_file.filename.endswith(".bin"):
        raise HTTPException(status_code=400, detail="The model BIN file must have .bin extension")

    try:
        await ModelService().add_model(model_name, xml_file, bin_file)
    except ModelAlreadyExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return ModelResponse(model_name=model_name)


@router.get("")
async def list_models() -> ModelsInfoResponse:  # TODO return list[schemas.model.Model]
    """Get information about available models"""
    model_service = ModelService()
    return ModelsInfoResponse(
        active_model=model_service.get_active_model_name(),
        available_models=model_service.get_available_model_names(),
    )


@router.get("/{model_id}")
async def get_model(model_id: Annotated[UUID, Depends(get_model_id)]) -> Model:
    """Get information about a specific model"""
    raise NotImplementedError


@router.patch("/{model_id}")
def update_model_metadata(
    model_id: Annotated[UUID, Depends(get_model_id)],
    model_metadata: Annotated[dict, Body(openapi_examples=UPDATE_MODEL_BODY_EXAMPLES)],
) -> Model:
    """Update the metadata of an existing model"""
    raise NotImplementedError


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(model_id: Annotated[UUID, Depends(get_model_id)]) -> None:
    """Delete a model"""
    raise NotImplementedError


# TODO remove this endpoint
@router.delete("/{model_name}", deprecated=True)
async def delete_model_by_name(model_name: str) -> None:
    """
    Delete a model by name

    NOTE: this endpoint will be removed; use `DELETE /api/models/{model_id}` instead
    """
    try:
        ModelService().remove_model(model_name)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# TODO remove this endpoint
@router.post("/{model_name}:activate", deprecated=True)
async def activate_model(model_name: str) -> ModelResponse:
    """
    Activate a model

    NOTE: this endpoint will be removed; use instead `PATCH /api/pipelines/{pipeline_id}` to change the active model
    """
    try:
        ModelService().activate_model(model_name)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return ModelResponse(model_name=model_name)
