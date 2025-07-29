"""Endpoints for managing pipelines"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, File, UploadFile, status
from fastapi.responses import FileResponse

from app.api.dependencies import get_pipeline_id
from app.schemas.pipeline import Pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/pipelines", tags=["Pipelines"])

CREATE_PIPELINE_BODY_DESCRIPTION = """
Configuration for the new pipeline. Requires the IDs of a source, sink, and model to be combined into a pipeline.
The fields 'source_id', 'sink_id', and 'model_id' can be set to `None` to only partially initialize the pipeline. 
"""
CREATE_PIPELINE_BODY_EXAMPLES = {
    "basic": {
        "summary": "Fully configured pipeline",
        "description": "Configuration for a basic pipeline with source, model, and sink",
        "value": {
            "name": "Production Pipeline",
            "source_id": "d2cbd8d0-17b8-463e-85a2-4aaed031674d",
            "sink_id": "b5787c06-964b-4097-8eca-238b8cf79fc8",
            "model_id": "b0feaabc-da2b-442e-9b3e-55c11c2c2ff2",
        },
    },
    "minimal": {
        "summary": "Partially configured pipeline",
        "description": "Pipeline with only a name (components can be assigned later)",
        "value": {
            "name": "Experimental Pipeline",
        },
    },
}

UPDATE_PIPELINE_BODY_DESCRIPTION = """
Partial pipeline configuration update. May contain any subset of fields including 'name', 'source_id', 
'sink_id', or 'model_id'. Fields not included in the request will remain unchanged.
"""
UPDATE_PIPELINE_BODY_EXAMPLES = {
    "switch_model": {
        "summary": "Switch active model",
        "description": "Change the active model for the pipeline",
        "value": {
            "model_id": "c1feaabc-da2b-442e-9b3e-55c11c2c2ff3",
        },
    },
    "reconfigure": {
        "summary": "Reconfigure pipeline",
        "description": "Change the name, source and sink of the pipeline",
        "value": {
            "name": "Updated Production Pipeline",
            "source_id": "e3cbd8d0-17b8-463e-85a2-4aaed031674e",
            "sink_id": "c6787c06-964b-4097-8eca-238b8cf79fc9",
        },
    },
}


@router.post("", status_code=status.HTTP_201_CREATED)
def create_pipeline(
    pipeline_config: Annotated[
        Pipeline, Body(description=CREATE_PIPELINE_BODY_DESCRIPTION, openapi_examples=CREATE_PIPELINE_BODY_EXAMPLES)
    ],
) -> Pipeline:
    """Create and configure a new pipeline"""
    raise NotImplementedError


@router.get("")
def list_pipelines() -> list[Pipeline]:
    """List the available pipelines"""
    raise NotImplementedError


@router.get("/{pipeline_id}")
def get_pipeline(pipeline_id: Annotated[UUID, Depends(get_pipeline_id)]) -> Pipeline:
    """Get info about a given pipeline"""
    raise NotImplementedError


@router.patch("/{pipeline_id}")
def update_pipeline(
    pipeline_id: Annotated[UUID, Depends(get_pipeline_id)],
    pipeline_config: Annotated[
        dict,
        Body(
            description=UPDATE_PIPELINE_BODY_DESCRIPTION,
            openapi_examples=UPDATE_PIPELINE_BODY_EXAMPLES,
        ),
    ],
) -> Pipeline:
    """Reconfigure an existing pipeline"""
    raise NotImplementedError


@router.post("/{pipeline_id}:enable", status_code=status.HTTP_204_NO_CONTENT)
def enable_pipeline(pipeline_id: Annotated[UUID, Depends(get_pipeline_id)]) -> None:
    """
    Activate a pipeline.
    The pipeline will start processing data from the source, run it through the model, and send results to the sink.
    """
    raise NotImplementedError


@router.post("/{pipeline_id}:disable", status_code=status.HTTP_204_NO_CONTENT)
def disable_pipeline(pipeline_id: Annotated[UUID, Depends(get_pipeline_id)]) -> None:
    """Stop a pipeline. The pipeline will become idle and it won't process any data until re-enabled."""
    raise NotImplementedError


@router.post(
    "/{pipeline_id}:export",
    response_class=FileResponse,
    responses={
        status.HTTP_200_OK: {
            "description": "Pipeline configuration exported as a ZIP file",
            "content": {
                "application/zip": {"schema": {"type": "string", "format": "binary"}},
            },
        }
    },
)
def export_pipeline(
    pipeline_id: Annotated[UUID, Depends(get_pipeline_id)], include_model: bool = False
) -> FileResponse:
    """Export a pipeline to file"""
    raise NotImplementedError


@router.post(":import", status_code=status.HTTP_204_NO_CONTENT)
def import_pipeline(
    zip_file: Annotated[
        UploadFile, File(description="ZIP file containing the pipeline configuration and optionally model binaries")
    ],
) -> None:
    """Import a pipeline from file"""
    raise NotImplementedError


@router.delete("/{pipeline_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_pipeline(pipeline_id: Annotated[UUID, Depends(get_pipeline_id)]) -> None:
    """Delete a pipeline. Pipelines must be first disabled (status must be idle) before deletion."""
    raise NotImplementedError
