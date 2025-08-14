from uuid import UUID

from app.db.schema import SinkDB, SourceDB
from app.repositories import SinkRepository, SourceRepository
from app.schemas import Sink, Source
from app.services.base import GenericPersistenceService, ResourceNotFoundError, ResourceType, ServiceConfig
from app.services.mappers import SinkMapper, SourceMapper
from app.utils import Singleton


class ConfigurationService(metaclass=Singleton):
    def __init__(self) -> None:
        self.source_service: GenericPersistenceService[Source, SourceDB, SourceRepository] = GenericPersistenceService(
            ServiceConfig(SourceRepository, SourceMapper, ResourceType.SOURCE)
        )
        self.sink_service: GenericPersistenceService[Sink, SinkDB, SinkRepository] = GenericPersistenceService(
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

    def update_source(self, source_id: UUID, partial_config: dict) -> Source:
        source = self.get_source_by_id(source_id)
        if not source:
            raise ResourceNotFoundError(ResourceType.SOURCE, str(source_id))
        return self.source_service.update(source, partial_config)

    def update_sink(self, sink_id: UUID, partial_config: dict) -> Sink:
        sink = self.get_sink_by_id(sink_id)
        if not sink:
            raise ResourceNotFoundError(ResourceType.SINK, str(sink_id))
        return self.sink_service.update(sink, partial_config)

    def delete_source_by_id(self, source_id: UUID) -> None:
        source = self.get_source_by_id(source_id)
        if not source:
            raise ResourceNotFoundError(ResourceType.SOURCE, str(source_id))
        self.source_service.delete_by_id(source_id)

    def delete_sink_by_id(self, sink_id: UUID) -> None:
        sink = self.get_sink_by_id(sink_id)
        if not sink:
            raise ResourceNotFoundError(ResourceType.SINK, str(sink_id))
        self.sink_service.delete_by_id(sink_id)
