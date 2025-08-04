"""Endpoints for managing pipeline sinks"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, File, UploadFile, status
from fastapi.openapi.models import Example
from fastapi.responses import FileResponse

from app.api.dependencies import get_sink_id
from app.schemas import Sink

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


@router.post("", status_code=status.HTTP_201_CREATED)
def create_sink(
    sink_config: Annotated[
        Sink, Body(description=CREATE_SINK_BODY_DESCRIPTION, openapi_examples=CREATE_SINK_BODY_EXAMPLES)
    ],
) -> Sink:
    """Create and configure a new sink"""
    raise NotImplementedError


@router.get("")
def list_sinks() -> list[Sink]:
    """List the available sinks"""
    raise NotImplementedError


@router.get("/{sink_id}")
def get_sink(sink_id: Annotated[UUID, Depends(get_sink_id)]) -> Sink:
    """Get info about a sink"""
    raise NotImplementedError


@router.patch("/{sink_id}")
def update_sink(
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
    raise NotImplementedError


@router.post(
    "/{sink_id}:export",
    response_class=FileResponse,
    responses={
        status.HTTP_200_OK: {
            "description": "Sink configuration exported as a YAML file",
            "content": {
                "application/x-yaml": {"schema": {"type": "string", "format": "binary"}},
            },
        }
    },
)
def export_sink(sink_id: Annotated[UUID, Depends(get_sink_id)]) -> FileResponse:
    """Export a sink to file"""
    raise NotImplementedError


@router.post(":import")
def import_sink(
    yaml_file: Annotated[UploadFile, File(description="YAML file containing the sink configuration")],
) -> Sink:
    """Import a sink from file"""
    raise NotImplementedError


@router.delete("/{sink_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_sink(sink_id: Annotated[UUID, Depends(get_sink_id)]) -> None:
    """Remove a sink"""
    raise NotImplementedError
