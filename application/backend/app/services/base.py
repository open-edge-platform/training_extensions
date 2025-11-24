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
    """
    Base class for services that require a managed database session.

    This class supports deferred database session initialization, allowing services
    to be instantiated without an immediate database connection. The session can be
    provided either at construction time or injected later via `set_db_session()`.

    This pattern is useful in scenarios where:
    - Services need to be created before database context is available
    - Database session management is handled externally (e.g., via session factories)
    - Services are used in contexts with different session lifecycle requirements

    Args:
        db_session: Optional database session to use immediately. If not provided,
            the session must be set later via `set_db_session()` or a factory must be provided.
        db_session_factory: Optional callable that returns a database session when invoked.
            Used as a fallback if no session is directly provided.

    Raises:
        RuntimeError: When accessing `db_session` property without a session or factory configured.

    Example:
        >>> # With immediate session
        >>> service = MyService(db_session=session)
        >>>
        >>> # With deferred session
        >>> service = MyService()
        >>> service.set_db_session(session)
        >>>
        >>> # With session factory
        >>> service = MyService(db_session_factory=lambda: get_session())
    """

    def __init__(
        self,
        db_session: Session | None = None,
        db_session_factory: Callable[[], Session] | None = None,
    ):
        self._db_session: Session | None = db_session
        self._db_session_factory = db_session_factory
        self._session_managed_services: list[BaseSessionManagedService] = []

    def set_db_session(self, db_session: Session) -> None:
        """Set the database session for the service."""
        self._db_session = db_session
        for service in self._session_managed_services:
            service.set_db_session(db_session)

    def register_managed_services(self, *services: "BaseSessionManagedService") -> None:
        """Register a child service that also requires session management."""
        self._session_managed_services.extend(services)

    @property
    def db_session(self) -> Session:
        if self._db_session is not None:
            return self._db_session
        if self._db_session_factory is not None:
            return self._db_session_factory()
        raise RuntimeError("No DB session available. Provide session or session factory.")
