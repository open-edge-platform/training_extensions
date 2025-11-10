# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from collections.abc import Callable
from enum import StrEnum

from sqlalchemy.orm import Session


class ResourceType(StrEnum):
    """Enumeration for resource types."""

    SOURCE = "Source"
    SINK = "Sink"
    MODEL = "Model"
    PIPELINE = "Pipeline"
    PROJECT = "Project"
    DATASET_ITEM = "DatasetItem"
    LABEL = "Label"


class ResourceError(Exception):
    """Base exception for resource-related errors."""

    def __init__(self, resource_type: ResourceType, resource_id: str, message: str):
        super().__init__(message)
        self.resource_type = resource_type
        self.resource_id = resource_id


class ResourceNotFoundError(ResourceError):
    """Exception raised when a resource is not found."""

    def __init__(self, resource_type: ResourceType, resource_id: str, message: str | None = None):
        msg = message or f"{resource_type} with ID {resource_id} not found."
        super().__init__(resource_type, resource_id, msg)


class ResourceInUseError(ResourceError):
    """Exception raised when trying to delete a resource that is currently in use."""

    def __init__(self, resource_type: ResourceType, resource_id: str, message: str | None = None):
        msg = message or f"{resource_type} with ID {resource_id} cannot be deleted because it is in use."
        super().__init__(resource_type, resource_id, msg)


class ResourceWithNameAlreadyExistsError(ResourceError):
    """Exception raised when a resource with the same name already exists."""

    def __init__(self, resource_type: ResourceType, resource_name: str, message: str | None = None):
        msg = message or f"{resource_type} with name '{resource_name}' already exists."
        super().__init__(resource_type, resource_name, msg)


class ResourceWithIdAlreadyExistsError(ResourceError):
    """Exception raised when a resource with the same ID already exists."""

    def __init__(self, resource_type: ResourceType, resource_id: str, message: str | None = None):
        msg = message or f"{resource_type} with ID '{resource_id}' already exists."
        super().__init__(resource_type, resource_id, msg)


class BaseSessionManagedService(ABC):
    """Base class for services that require a managed database session."""

    def __init__(
        self,
        db_session: Session | None = None,
        db_session_factory: Callable[[], Session] | None = None,
    ):
        self._db_session: Session | None = db_session
        self._db_session_factory = db_session_factory

    def set_db_session(self, db_session: Session) -> None:
        """Set the database session for the service."""
        self._db_session = db_session

    @property
    def db_session(self) -> Session:
        if self._db_session is not None:
            return self._db_session
        if self._db_session_factory is not None:
            return self._db_session_factory()
        raise RuntimeError("No DB session available. Provide session or session factory.")
