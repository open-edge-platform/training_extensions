from enum import Enum
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from app.db import get_db_session
from app.repositories import SourceRepository
from app.schemas import Source
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


class ConfigurationService(metaclass=Singleton):
    @staticmethod
    def list_sources() -> list[Source]:
        with get_db_session() as db:
            source_repo = SourceRepository(db)
            return [SourceMapper.to_schema(o) for o in source_repo.list_all()]

    @staticmethod
    def get_source_by_id(source_id: UUID) -> Source | None:
        with get_db_session() as db:
            source_repo = SourceRepository(db)
            source_db = source_repo.get_by_id(str(source_id))
            if source_db is None:
                return None
            return SourceMapper.to_schema(source_db)

    @staticmethod
    def create_source(source: Source) -> Source:
        with get_db_session() as db:
            source_repo = SourceRepository(db)
            source_db = SourceMapper.from_schema(source)
            source_repo.save(source_db)
            db.commit()
            return SourceMapper.to_schema(source_db)

    @staticmethod
    def update_source(source: Source, partial_config: dict) -> Source:
        with get_db_session() as db:
            source_repo = SourceRepository(db)
            update = source.model_copy(update=partial_config)
            source_db = SourceMapper.from_schema(update)
            source_repo.update(source_db)
            db.commit()
            return SourceMapper.to_schema(source_db)

    @staticmethod
    def delete_source_by_id(source_id: UUID) -> None:
        try:
            with get_db_session() as db:
                source_repo = SourceRepository(db)
                source_repo.delete(str(source_id))
                db.commit()
        except IntegrityError:
            raise ResourceInUseError(ResourceType.SOURCE, str(source_id))
