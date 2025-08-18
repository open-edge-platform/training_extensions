"""Endpoints for managing pipeline sinks"""

import logging
from typing import Annotated
from uuid import UUID

import yaml
from fastapi import APIRouter, Body, Depends, File, UploadFile, status
from fastapi.exceptions import HTTPException
from fastapi.openapi.models import Example
from fastapi.responses import FileResponse, Response

from app.api.dependencies import get_sink_id
from app.schemas import Sink, SinkType
from app.schemas.sink import SinkAdapter
from app.services import ConfigurationService, ResourceInUseError, ResourceNotFoundError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sinks", tags=["Sinks"])

CREATE_SINK_BODY_DESCRIPTION = """
Configuration for the new sink. The exact list of fields that can be configured depends on the sink type.
"""
CREATE_SINK_BODY_EXAMPLES = {
    "folder": Example(
        summary="Folder sink",
        description="Configuration for a local filesystem folder sink",
        value={
            "sink_type": "folder",
            "name": "My Output Folder",
            "folder_path": "/path/to/output",
            "output_formats": ["image_with_predictions"],
            "rate_limit": 0.2,
        },
    ),
    "mqtt": Example(
        summary="MQTT sink",
        description="Configuration for an MQTT message broker sink",
        value={
            "sink_type": "mqtt",
            "name": "Local MQTT Broker",
            "broker_host": "localhost",
            "broker_port": 1883,
            "topic": "predictions",
            "output_formats": ["predictions"],
        },
    ),
}

UPDATE_SINK_BODY_DESCRIPTION = """
Partial sink configuration update. May contain any subset of fields from the respective sink type
(e.g., 'broker_host' and 'broker_port' for MQTT; 'folder_path' for folder sinks).
Fields not included in the request will remain unchanged. The 'sink_type' field cannot be changed.
"""
UPDATE_SINK_BODY_EXAMPLES = {
    "folder": Example(
        summary="Update folder sink",
        description="Change the output path for a folder sink",
        value={
            "folder_path": "/new/output/directory",
        },
    ),
    "mqtt": Example(
        summary="Update MQTT sink",
        description="Change the topic for an MQTT sink",
        value={
            "topic": "new_predictions_topic",
        },
    ),
}


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {"description": "Sink created", "model": Sink},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid sink ID or request body"},
    },
)
async def create_sink(
    sink_config: Annotated[
        Sink, Body(description=CREATE_SINK_BODY_DESCRIPTION, openapi_examples=CREATE_SINK_BODY_EXAMPLES)
    ],
) -> Sink:
    """Create and configure a new sink"""
    if sink_config.sink_type == SinkType.DISCONNECTED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The sink with sink_type=DISCONNECTED cannot be created",
        )

    return ConfigurationService().create_sink(sink_config)


@router.get(
    "",
    responses={
        status.HTTP_200_OK: {"description": "List of available sink configurations", "model": list[Sink]},
    },
)
async def list_sinks() -> list[Sink]:
    """List the available sinks"""
    return ConfigurationService().list_sinks()


@router.get(
    "/{sink_id}",
    responses={
        status.HTTP_200_OK: {"description": "Sink found", "model": Sink},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid sink ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Sink not found"},
    },
)
async def get_sink(sink_id: Annotated[UUID, Depends(get_sink_id)]) -> Sink:
    """Get info about a sink"""
    try:
        return ConfigurationService().get_sink_by_id(sink_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.patch(
    "/{sink_id}",
    responses={
        status.HTTP_200_OK: {"description": "Sink successfully updated", "model": Sink},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid sink ID or request body"},
        status.HTTP_404_NOT_FOUND: {"description": "Sink not found"},
    },
)
async def update_sink(
    sink_id: Annotated[UUID, Depends(get_sink_id)],
    sink_config: Annotated[
        dict,
        Body(
            description=(
                "Partial sink configuration update. "
                "May contain any subset of fields from the respective sink type "
                "(e.g., 'broker_host' and 'broker_port' for MQTT; 'output_path' for folder sinks). "
                "Fields not included in the request will remain unchanged. "
                "The 'sink_type' field cannot be changed."
            ),
            openapi_examples=UPDATE_SINK_BODY_EXAMPLES,
        ),
    ],
) -> Sink:
    """Reconfigure an existing sink"""
    if "sink_type" in sink_config:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The 'sink_type' field cannot be changed")
    try:
        return ConfigurationService().update_sink(sink_id, sink_config)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post(
    "/{sink_id}:export",
    response_class=FileResponse,
    responses={
        status.HTTP_200_OK: {
            "description": "Sink configuration exported as a YAML file",
            "content": {
                "application/x-yaml": {"schema": {"type": "string", "format": "binary"}},
            },
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid sink ID or request body"},
        status.HTTP_404_NOT_FOUND: {"description": "Sink not found"},
    },
)
async def export_sink(sink_id: Annotated[UUID, Depends(get_sink_id)]) -> Response:
    """Export a sink to file"""
    sink = ConfigurationService().get_sink_by_id(sink_id)
    if not sink:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sink with ID {sink_id} not found",
        )

    yaml_content = yaml.safe_dump(sink.model_dump(mode="json", exclude={"id"}))

    return Response(
        content=yaml_content.encode("utf-8"),
        media_type="application/x-yaml",
        headers={"Content-Disposition": f"attachment; filename=sink_{sink_id}.yaml"},
    )


@router.post(
    ":import",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {"description": "Sink imported successfully", "model": Sink},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid YAML format or sink type is DISCONNECTED"},
    },
)
async def import_sink(
    yaml_file: Annotated[UploadFile, File(description="YAML file containing the sink configuration")],
) -> Sink:
    """Import a sink from file"""
    try:
        yaml_content = await yaml_file.read()
        sink_data = yaml.safe_load(yaml_content)

        sink_config = SinkAdapter.validate_python(sink_data)
        if sink_config.sink_type == SinkType.DISCONNECTED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The sink with sink_type=DISCONNECTED cannot be imported",
            )
        return ConfigurationService().create_sink(sink_config)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid YAML format: {str(e)}")


@router.delete(
    "/{sink_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {
            "description": "Sink configuration successfully deleted",
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid sink ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Sink not found"},
        status.HTTP_409_CONFLICT: {"description": "Sink is used by at least one pipeline"},
    },
)
async def delete_sink(sink_id: Annotated[UUID, Depends(get_sink_id)]) -> None:
    """Remove a sink"""
    try:
        ConfigurationService().delete_sink_by_id(sink_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ResourceInUseError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
