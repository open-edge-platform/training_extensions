from pydantic import TypeAdapter

from app.db.schema import SourceDB
from app.schemas.source import Source, SourceAdapter, SourceType


class SourceMapper:
    """Mapper for Source model <-> Source schema conversions."""

    # Define fields to exclude from config_data (common fields)
    _COMMON_FIELDS: set[str] = {"id", "name", "source_type", "enabled", "created_at", "updated_at"}

    # Create TypeAdapter for better serialization performance
    _SOURCE_ADAPTER: TypeAdapter[Source] = TypeAdapter(Source)

    @staticmethod
    def to_schema(source_db: SourceDB) -> Source:
        """Convert Source model to Source schema."""

        config_data = source_db.config_data or {}
        return SourceAdapter.validate_python(
            {
                "id": source_db.id,
                "name": source_db.name,
                "source_type": SourceType(source_db.source_type),
                **config_data,
            }
        )

    @staticmethod
    def from_schema(source: Source) -> SourceDB:
        """Convert Source schema to Source model."""
        if source is None:
            raise ValueError("Source config cannot be None")

        source_dict = SourceMapper._SOURCE_ADAPTER.dump_python(
            source, exclude=SourceMapper._COMMON_FIELDS, exclude_none=True
        )

        return SourceDB(
            id=str(source.id),
            name=source.name,
            source_type=source.source_type.value,
            config_data=source_dict,
        )
