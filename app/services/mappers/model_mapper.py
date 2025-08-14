from uuid import UUID

from app.db.schema import ModelDB
from app.schemas.model import Model, ModelFormat


class ModelMapper:
    """Mapper for Model schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(model_db: ModelDB) -> Model:
        """Convert Model db entity to schema."""

        return Model(
            id=UUID(model_db.id),
            name=model_db.name,
            format=ModelFormat(model_db.format),
        )

    @staticmethod
    def from_schema(model: Model) -> ModelDB:
        """Convert Model schema to db model."""

        return ModelDB(
            id=str(model.id),
            name=model.name,
            format=model.format,
        )
