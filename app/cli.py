"""Command line interface for database operations"""

import logging
import sys

import click

from app.db import get_db_session, migration_manager
from app.db.schema import ModelDB, PipelineDB, SinkDB, SourceDB
from app.schemas import ModelFormat, OutputFormat, SinkType, SourceType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """GETI Edge CLI"""


@cli.command()
def init_db() -> None:
    """Initialize database with migrations"""
    click.echo("Initializing database...")

    if migration_manager.initialize_database():
        click.echo("✓ Database initialized successfully!")
        sys.exit(0)
    else:
        click.echo("✗ Database initialization failed!")
        sys.exit(1)


@cli.command()
def migrate() -> None:
    """Run database migrations"""
    click.echo("Running database migrations...")

    if migration_manager.run_migrations():
        click.echo("✓ Migrations completed successfully!")
        sys.exit(0)
    else:
        click.echo("✗ Migration failed!")
        sys.exit(1)


@cli.command()
def check_db() -> None:
    """Check database status"""
    click.echo("Checking database status...")

    # Check connection
    if not migration_manager.check_connection():
        click.echo("✗ Cannot connect to database")
        sys.exit(1)

    click.echo("✓ Database connection OK")

    # Check migration status
    needs_migration, status = migration_manager.check_migration_status()
    click.echo(f"Migration status: {status}")

    if needs_migration:
        click.echo("⚠ Database needs migration")
        sys.exit(2)
    else:
        click.echo("✓ Database is up to date")
        sys.exit(0)


@cli.command()
@click.option("--with-model", default=False)
def seed(with_model: bool) -> None:
    """Seed the database with test data."""
    # TODO: use APIs when ready
    # Currently, the app needs to be restarted since it doesn't track direct DB changes
    with get_db_session() as db:
        source = SourceDB(
            name="Default source",
            source_type=SourceType.VIDEO_FILE.value,
            config_data={"video_path": "data/media/video.mp4"},
        )
        db.add(source)
        sink = SinkDB(
            name="Default sink",
            sink_type=SinkType.FOLDER.value,
            rate_limit=0.2,
            output_formats=[OutputFormat.IMAGE_ORIGINAL, OutputFormat.IMAGE_WITH_PREDICTIONS, OutputFormat.PREDICTIONS],
            config_data={"folder_path": "data/output"},
        )
        db.add(sink)
        model = None
        if with_model:
            model = ModelDB(
                name="card-detection-ssd",
                format=ModelFormat.OPENVINO.value,
            )
            db.add(model)
        db.flush()
        pipeline = PipelineDB(
            source_id=source.id,
            sink_id=sink.id,
            model_id=model.id if model else None,
            name="Video Processing Pipeline",
            description="Pipeline for processing video files",
            is_running=True,
        )
        db.add(pipeline)
        db.commit()
    click.echo("✓ Seeding successful!")


@cli.command()
def clean_db() -> None:
    """Remove all data from the database (clean but don't drop tables)."""
    with get_db_session() as db:
        db.query(PipelineDB).delete()
        db.query(ModelDB).delete()
        db.query(SinkDB).delete()
        db.query(SourceDB).delete()
        db.commit()
    click.echo("✓ Database cleaned successfully!")


if __name__ == "__main__":
    cli()
