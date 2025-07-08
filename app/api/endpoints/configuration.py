import logging

from fastapi import APIRouter, HTTPException

from app.schemas.configuration import InputConfig, OutputConfig
from app.services import ConfigurationService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.get("/inputs")
async def get_input_configuration() -> InputConfig:
    """Get the current input stream configuration."""
    config_service = ConfigurationService()
    return config_service.get_input_config()


@router.get("/outputs")
async def get_output_configurations() -> list[OutputConfig]:
    """Get the current output configurations."""
    config_service = ConfigurationService()
    return config_service.get_output_config()


@router.post("/inputs")
async def configure_input_stream(input_config: InputConfig) -> None:
    """Configure the input stream for the application."""
    config_service = ConfigurationService()
    try:
        config_service.set_input_config(input_config)
    except Exception as e:
        logger.exception("Failed to update input configuration")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/outputs")
async def configure_outputs(outputs: list[OutputConfig]) -> None:
    """Configure the destination(s) for the application output."""
    config_service = ConfigurationService()
    try:
        config_service.set_output_config(outputs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
