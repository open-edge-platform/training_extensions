from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Protocol, TypeVar
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError

from app.db import get_db_session
from app.db.schema import Base, SinkDB, SourceDB
from app.repositories import SinkRepository, SourceRepository
from app.repositories.base import BaseRepository
from app.schemas import Sink, Source
from app.services.mappers.sink_mapper import SinkMapper
from app.services.mappers.source_mapper import SourceMapper
from app.utils import Singleton


class ResourceType(str, Enum):
    """Enumeration for resource types."""

    SOURCE = "Source"
    SINK = "Sink"


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
class ServiceConfig(Generic[R]):
    repository_class: type[R]
    mapper_class: MapperProtocol
    resource_type: ResourceType


class GenericConfigurationService(Generic[S, D, R]):
    def __init__(self, config: ServiceConfig[R]) -> None:
        self.config = config

    @contextmanager
    def _get_repo(self) -> Generator[R, None, None]:
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


class ConfigurationService(metaclass=Singleton):
    def __init__(self) -> None:
        self.source_service: GenericConfigurationService[Source, SourceDB, SourceRepository] = (
            GenericConfigurationService(ServiceConfig(SourceRepository, SourceMapper, ResourceType.SOURCE))
        )
        self.sink_service: GenericConfigurationService[Sink, SinkDB, SinkRepository] = GenericConfigurationService(
            ServiceConfig(SinkRepository, SinkMapper, ResourceType.SINK)
        )

    def list_sources(self) -> list[Source]:
        return self.source_service.list_all()

    def list_sinks(self) -> list[Sink]:
        return self.sink_service.list_all()

    def get_source_by_id(self, source_id: UUID) -> Source | None:
        return self.source_service.get_by_id(source_id)

    def get_sink_by_id(self, sink_id: UUID) -> Sink | None:
        return self.sink_service.get_by_id(sink_id)

    def create_source(self, source: Source) -> Source:
        return self.source_service.create(source)

    def create_sink(self, sink: Sink) -> Sink:
        return self.sink_service.create(sink)

    def update_source(self, source: Source, partial_config: dict) -> Source:
        return self.source_service.update(source, partial_config)

    def update_sink(self, sink: Sink, partial_config: dict) -> Sink:
        return self.sink_service.update(sink, partial_config)

    def delete_source_by_id(self, source_id: UUID) -> None:
        self.source_service.delete_by_id(source_id)

    def delete_sink_by_id(self, sink_id: UUID) -> None:
        self.sink_service.delete_by_id(sink_id)
