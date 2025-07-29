from uuid import UUID

from fastapi import HTTPException


def is_valid_uuid(identifier: str) -> bool:
    """
    Check if a given string identifier is formatted as a valid UUID

    :param identifier: String to check
    :return: True if valid UUID, False otherwise
    """
    try:
        UUID(identifier)
    except ValueError:
        return False
    return True


def get_source_id(source_id: str) -> UUID:
    """Initializes and validates a source ID"""
    if not is_valid_uuid(source_id):
        raise HTTPException(status_code=400, detail="Invalid source ID")
    return UUID(source_id)


def get_sink_id(sink_id: str) -> UUID:
    """Initializes and validates a sink ID"""
    if not is_valid_uuid(sink_id):
        raise HTTPException(status_code=400, detail="Invalid sink ID")
    return UUID(sink_id)


def get_model_id(model_id: str) -> UUID:
    """Initializes and validates a model ID"""
    if not is_valid_uuid(model_id):
        raise HTTPException(status_code=400, detail="Invalid model ID")
    return UUID(model_id)


def get_pipeline_id(pipeline_id: str) -> UUID:
    """Initializes and validates a pipeline ID"""
    if not is_valid_uuid(pipeline_id):
        raise HTTPException(status_code=400, detail="Invalid pipeline ID")
    return UUID(pipeline_id)
