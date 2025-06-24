from typing import Annotated, Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel
from services.model_service import (
    ModelAlreadyExistsError,
    ModelNotFoundError,
    ModelService,
)

router = APIRouter(prefix="/api")


class ModelResponse(BaseModel):
    model_name: str


class ModelsInfoResponse(BaseModel):
    active_model: str | None
    available_models: list[str]


@router.post("/models")
async def add_model(
    model_name: Annotated[str, Query(description="Name for the model files")],
    xml_file: Annotated[UploadFile, File()],
    bin_file: Annotated[UploadFile, File()],
) -> ModelResponse:
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


@router.get("/models")
async def get_models() -> ModelsInfoResponse:
    """Get information about available models"""
    model_service = ModelService()
    return ModelsInfoResponse(
        active_model=model_service.get_active_model_name(),
        available_models=model_service.get_available_model_names(),
    )


@router.delete("/models/{model_name}", response_model=None)
async def delete_model(model_name: str) -> None:
    """Delete a model"""
    try:
        ModelService().remove_model(model_name)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/models/{model_name}:activate")
async def activate_model(model_name: str) -> ModelResponse:
    """Activate a model"""
    try:
        ModelService().activate_model(model_name)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return ModelResponse(model_name=model_name)
