import logging

from fastapi import APIRouter, HTTPException

from app.schemas.configuration import Sink, Source
from app.services import ConfigurationService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.get("/inputs", deprecated=True)
async def get_source_config() -> Source:
    """
    Get the current input stream configuration.

    NOTE: this endpoint will be removed; you should use instead:
        - `GET /api/sources` to list the sources
        - `GET /api/pipelines/{pipeline_id}` to find the source used in a given pipeline
    """
    config_service = ConfigurationService()
    return config_service.get_source_config()


@router.get("/outputs", deprecated=True)
async def get_sink_config() -> Sink:
    """
    Get the current output configurations.

    NOTE: this endpoint will be removed; you should use instead:
        - `GET /api/sinks` to list the sinks
        - `GET /api/pipelines/{pipeline_id}` to find the sink used in a given pipeline
    """
    config_service = ConfigurationService()
    return config_service.get_sink_config()


@router.post("/inputs", deprecated=True)
async def configure_source(source: Source) -> None:
    """
    Configure the input stream for the application.

    NOTE: this endpoint will be removed; you should use instead:
        - `POST /api/sources` to create a new source
        - `PATCH /api/sources/{source_id}` to update an existing source
        - `PATCH /api/pipelines/{pipeline_id}` to change the source used in a pipeline
    """
    config_service = ConfigurationService()
    try:
        config_service.set_source_config(source)
    except Exception as e:
        logger.exception("Failed to update input configuration")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/outputs", deprecated=True)
async def configure_sink(sink: Sink) -> None:
    """
    Configure the destination(s) for the application output.

    NOTE: this endpoint will be removed; you should use instead:
        - `POST /api/sinks` to create a new sink
        - `PATCH /api/sinks/{sink_id}` to update an existing sink
        - `PATCH /api/pipelines/{pipeline_id}` to change the sink used in a pipeline
    """
    config_service = ConfigurationService()
    try:
        config_service.set_sink_config(sink)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
