from app.db.schema import PipelineDB
from app.schemas import Pipeline, PipelineStatus


class PipelineMapper:
    """Mapper for Pipeline schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(pipeline_db: PipelineDB) -> Pipeline:
        """Convert Pipeline db entity to schema."""

        return Pipeline.model_validate(pipeline_db, from_attributes=True)

    @staticmethod
    def from_schema(pipeline: Pipeline) -> PipelineDB:
        """Convert Pipeline schema to db model."""

        obj = pipeline.model_dump(mode="json")
        obj["is_running"] = obj.pop("status") == PipelineStatus.RUNNING
        return PipelineDB(**obj)
