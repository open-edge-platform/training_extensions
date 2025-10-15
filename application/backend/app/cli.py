# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Command line interface for interacting with the Geti Tune application."""

import logging
import sys
from datetime import datetime, timedelta

import click

from app.db import MigrationManager, get_db_session
from app.db.schema import DatasetItemDB, LabelDB, ModelRevisionDB, PipelineDB, ProjectDB, SinkDB, SourceDB
from app.schemas import DisconnectedSinkConfig, DisconnectedSourceConfig, OutputFormat, SinkType, SourceType
from app.schemas.model import TrainingStatus
from app.schemas.pipeline import FixedRateDataCollectionPolicy
from app.schemas.project import TaskType
from app.settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()
migration_manager = MigrationManager(settings)


@click.group()
def cli() -> None:
    """Geti Tune CLI"""


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
    # If the app is running, it needs to be restarted since it doesn't track direct DB changes
    # Fixed IDs are used to ensure consistency in tests
    click.echo("Seeding database with test data...")
    project_id = "9d6af8e8-6017-4ebe-9126-33aae739c5fa"
    with get_db_session() as db:
        project = ProjectDB(
            id=project_id,
            name="Test Project",
            task_type=TaskType.DETECTION,
            exclusive_labels=True,
        )
        db.add(project)
        db.flush()
        labels = [
            LabelDB(project_id=project_id, name="Clubs", color="#2d6311", hotkey="c"),
            LabelDB(project_id=project_id, name="Diamonds", color="#baa3b3", hotkey="d"),
            LabelDB(project_id=project_id, name="Spades", color="#000702", hotkey="s"),
            LabelDB(project_id=project_id, name="Hearts", color="#1f016b", hotkey="h"),
            LabelDB(project_id=project_id, name="No_object", color="#565a84", hotkey="n"),
        ]
        db.add_all(labels)
        db.flush()

        # Create default disconnected source and sink
        disconnected_source_cfg = DisconnectedSourceConfig()
        disconnected_source = SourceDB(
            id="00000000-0000-0000-0000-000000000000",
            name=disconnected_source_cfg.name,
            source_type=disconnected_source_cfg.source_type,
            config_data={},
        )
        disconnected_sink_cfg = DisconnectedSinkConfig()
        disconnected_sink = SinkDB(
            id="00000000-0000-0000-0000-000000000000",
            name=disconnected_sink_cfg.name,
            sink_type=disconnected_sink_cfg.sink_type,
            output_formats=[],
            config_data={},
        )
        db.add_all([disconnected_source, disconnected_sink])

        pipeline = PipelineDB(project_id=project.id)
        pipeline.source = SourceDB(
            id="f6b1ac22-e36c-4b36-9a23-62b0881e4223",
            name="Video Source",
            source_type=SourceType.VIDEO_FILE,
            config_data={"video_path": "data/media/video.mp4"},
        )
        pipeline.sink = SinkDB(
            id="6ee0c080-c7d9-4438-a7d2-067fd395eecf",
            name="Folder Sink",
            sink_type=SinkType.FOLDER,
            rate_limit=0.2,
            output_formats=[OutputFormat.IMAGE_ORIGINAL, OutputFormat.IMAGE_WITH_PREDICTIONS, OutputFormat.PREDICTIONS],
            config_data={"folder_path": "data/output"},
        )
        pipeline.data_collection_policies = [FixedRateDataCollectionPolicy(rate=0.1).model_dump(mode="json")]
        if with_model:
            pipeline.model_revision = ModelRevisionDB(
                id="977eeb18-eaac-449d-bc80-e340fbe052ad",
                project_id=project.id,
                architecture="Object_Detection_SSD",
                training_status=TrainingStatus.SUCCESSFUL,
                training_started_at=datetime.now() - timedelta(hours=24),
                training_finished_at=datetime.now() - timedelta(hours=23),
                training_configuration={},
                label_schema_revision={},
            )
            pipeline.is_running = True
        db.add(pipeline)
        db.commit()
    click.echo("✓ Seeding successful!")


@cli.command()
def clean_db() -> None:
    """Remove all data from the database (clean but don't drop tables)."""
    with get_db_session() as db:
        db.query(DatasetItemDB).delete()
        db.query(ProjectDB).delete()
        db.query(ModelRevisionDB).delete()
        db.query(SinkDB).delete()
        db.query(SourceDB).delete()
        db.commit()
    click.echo("✓ Database cleaned successfully!")


@cli.command()
@click.option("--target-path", default="docs/openapi.json")
def gen_api(target_path: str) -> None:
    """Generate OpenAPI specification JSON file."""
    # Importing create_openapi imports threading which is slow. Importing here to not slow down other cli commands.
    from app.create_openapi import create_openapi

    try:
        create_openapi(target_path=target_path)
        click.echo("✓ OpenAPI specification generated successfully!")
    except Exception as e:
        click.echo(f"✗ Failed to generate OpenAPI specification: {e}")
        sys.exit(1)
    click.echo("Waiting for threading to finish...")


if __name__ == "__main__":
    cli()
