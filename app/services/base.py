from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, Protocol, TypeVar
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError

from app.db import get_db_session
from app.db.schema import Base
from app.repositories.base import BaseRepository


class ResourceType(StrEnum):
    """Enumeration for resource types."""

    SOURCE = "Source"
    SINK = "Sink"
    MODEL = "Model"


class ResourceNotFoundError(Exception):
    """Exception raised when a resource is not found."""

    def __init__(self, resource_type: ResourceType, resource_id: str, message: str | None = None):
        msg = message or f"{resource_type} with ID {resource_id} not found."
        super().__init__(msg)
        self.resource_type = resource_type
        self.resource_id = resource_id


class ResourceInUseError(Exception):
    """Exception raised when trying to delete a resource that is currently in use."""

    def __init__(self, resource_type: ResourceType, resource_id: str, message: str | None = None):
        msg = message or f"{resource_type} with ID {resource_id} cannot be deleted because it is in use."
        super().__init__(msg)
        self.resource_type = resource_type
        self.resource_id = resource_id


S = TypeVar("S", bound=BaseModel)  # Schema type e.g. Source or Sink
D = TypeVar("D", bound=Base)  # DB model type e.g. SourceDB or SinkDB
R = TypeVar("R", bound=BaseRepository)  # Repository type


class MapperProtocol(Protocol[S, D]):
    """Protocol for mapper classes."""

    @staticmethod
    def to_schema(db_model: D) -> S: ...

    @staticmethod
    def from_schema(schema: S) -> D: ...


@dataclass(frozen=True)
class ServiceConfig(Generic[R]):  # noqa: UP046
    repository_class: type[R]
    mapper_class: MapperProtocol
    resource_type: ResourceType


class GenericPersistenceService(Generic[S, D, R]):  # noqa: UP046
    def __init__(self, config: ServiceConfig[R]) -> None:
        self.config = config

    @contextmanager
    def _get_repo(self) -> Generator[R]:
        with get_db_session() as db:
            repo = self.config.repository_class(db)  # type: ignore[call-arg]
            yield repo
            db.commit()

    def list_all(self) -> list[S]:
        with self._get_repo() as repo:
            return [self.config.mapper_class.to_schema(o) for o in repo.list_all()]

    def get_by_id(self, item_id: UUID) -> S | None:
        with self._get_repo() as repo:
            item_db = repo.get_by_id(str(item_id))
            return self.config.mapper_class.to_schema(item_db) if item_db else None

    def create(self, item: S) -> S:
        with self._get_repo() as repo:
            item_db = self.config.mapper_class.from_schema(item)
            repo.save(item_db)
            return self.config.mapper_class.to_schema(item_db)

    def update(self, item: S, partial_config: dict) -> S:
        with self._get_repo() as repo:
            update = item.model_copy(update=partial_config)
            item_db = self.config.mapper_class.from_schema(update)
            repo.update(item_db)
            return self.config.mapper_class.to_schema(item_db)

    def delete_by_id(self, item_id: UUID) -> None:
        try:
            with self._get_repo() as repo:
                repo.delete(str(item_id))
        except IntegrityError:
            raise ResourceInUseError(self.config.resource_type, str(item_id))
