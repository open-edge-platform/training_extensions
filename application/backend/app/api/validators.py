# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException
from starlette import status


def validate_uuid_param(value: str, param_name: str) -> UUID:
    """
    Validate and convert a string to a UUID object.

    This function validates that the provided string value is a valid UUID format
    and converts it to a UUID object. If validation fails, it raises an HTTPException
    with a descriptive error message.

    Args:
        value: The string value to validate as a UUID.
        param_name: The name of the parameter for error messaging. Defaults to "ID".

    Returns:
        A UUID object if validation is successful.

    Raises:
        HTTPException: A 400 Bad Request error if the value is not a valid UUID format.

    Examples:
        >>> validate_uuid_param("550e8400-e29b-41d4-a716-446655440000", "user_id")
        UUID('550e8400-e29b-41d4-a716-446655440000')

        >>> validate_uuid_param("invalid-uuid", "user_id")
        HTTPException: Invalid user_id: 'invalid-uuid' must be a valid UUID format

    """
    try:
        return UUID(value)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {param_name}: '{value}' must be a valid UUID format.",
        )


JobID = Annotated[UUID, Depends(lambda job_id: validate_uuid_param(job_id, "job_id"))]
SourceID = Annotated[UUID, Depends(lambda source_id: validate_uuid_param(source_id, "source_id"))]
SinkID = Annotated[UUID, Depends(lambda sink_id: validate_uuid_param(sink_id, "sink_id"))]
ProjectID = Annotated[UUID, Depends(lambda project_id: validate_uuid_param(project_id, "project_id"))]
ModelID = Annotated[UUID, Depends(lambda model_id: validate_uuid_param(model_id, "model_id"))]
DatasetItemID = Annotated[
    UUID, Depends(lambda dataset_item_id: validate_uuid_param(dataset_item_id, "dataset_item_id"))
]
DatasetRevisionID = Annotated[
    UUID, Depends(lambda dataset_revision_id: validate_uuid_param(dataset_revision_id, "dataset_revision_id"))
]
