"""seed data

Revision ID: 6bcfb74ac696
Revises: b1087d3a75ca
Create Date: 2025-07-28 16:02:26.286510

"""

from collections.abc import Sequence
from uuid import uuid4

from alembic import op
from sqlalchemy import MetaData

from app.schemas.configuration.input_config import SourceType
from app.schemas.configuration.output_config import SinkType
from app.schemas.model import ModelFormat

# revision identifiers, used by Alembic.
revision: str = "6bcfb74ac696"
down_revision: str | Sequence[str] | None = "b1087d3a75ca"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add data."""
    # Get table references
    metadata = MetaData()
    metadata.reflect(bind=op.get_bind())

    sources_table = metadata.tables["sources"]
    sinks_table = metadata.tables["sinks"]
    models_table = metadata.tables["models"]
    pipelines_table = metadata.tables["pipelines"]

    source_id = str(uuid4())
    op.bulk_insert(
        sources_table,
        [
            {
                "id": source_id,
                "source_type": SourceType.VIDEO_FILE.value,
                "config_data": {"video_path": "data/media/video.mp4"},
            }
        ],
    )
    sink_id = str(uuid4())
    op.bulk_insert(
        sinks_table,
        [
            {
                "id": sink_id,
                "sink_type": SinkType.FOLDER.value,
                "rate_limit": 0.2,
                "output_formats": ["image_original", "image_with_predictions", "predictions"],
                "config_data": {"folder_path": "data/output"},
            }
        ],
    )
    model_id = str(uuid4())
    op.bulk_insert(
        models_table,
        [
            {
                "id": model_id,
                "name": "card-detection-ssd",
                "format": ModelFormat.OPENVINO.value,
            }
        ],
    )
    op.bulk_insert(
        pipelines_table,
        [
            {
                "id": str(uuid4()),
                "source_id": source_id,
                "sink_id": sink_id,
                "model_id": model_id,
                "name": "Video Processing Pipeline",
                "description": "Pipeline for processing video files",
                "is_running": True,
            }
        ],
    )


def downgrade() -> None:
    """Remove data."""
    # Clear all data from tables
    op.execute("DELETE FROM pipelines")
    op.execute("DELETE FROM models")
    op.execute("DELETE FROM sinks")
    op.execute("DELETE FROM sources")
