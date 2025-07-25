import logging

from fastapi import APIRouter, HTTPException

from app.schemas.configuration import Sink, Source
from app.services import ConfigurationService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.get("/inputs")
async def get_source_config() -> Source:
    """Get the current input stream configuration."""
    config_service = ConfigurationService()
    return config_service.get_source_config()


@router.get("/outputs")
async def get_sink_config() -> Sink:
    """Get the current output configurations."""
    config_service = ConfigurationService()
    return config_service.get_sink_config()


@router.post("/inputs")
async def configure_source(source: Source) -> None:
    """Configure the input stream for the application."""
    config_service = ConfigurationService()
    try:
        config_service.set_source_config(source)
    except Exception as e:
        logger.exception("Failed to update input configuration")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/outputs")
async def configure_sink(sink: Sink) -> None:
    """Configure the destination(s) for the application output."""
    config_service = ConfigurationService()
    try:
        config_service.set_sink_config(sink)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
