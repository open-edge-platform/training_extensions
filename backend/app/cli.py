# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Command line interface for interacting with the Geti Tune application."""

import logging
import sys

import click

from app.db import get_db_session, migration_manager
from app.db.schema import DatasetItemDB, ModelDB, PipelineDB, ProjectDB, SinkDB, SourceDB
from app.schemas import ModelFormat, OutputFormat, SinkType, SourceType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
@click.option("--model-name", default="card-detection-ssd")
def seed(with_model: bool, model_name: str) -> None:
    """Seed the database with test data."""
    # If the app is running, it needs to be restarted since it doesn't track direct DB changes
    # Fixed IDs are used to ensure consistency in tests
    click.echo("Seeding database with test data...")
    with get_db_session() as db:
        project = ProjectDB(
            id="9d6af8e8-6017-4ebe-9126-33aae739c5fa",
            name="Test Project",
            task_type="detection",
            exclusive_labels=True,
            labels=["card", "person"],
        )
        pipeline = PipelineDB()
        project.pipeline = pipeline
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
        if with_model:
            pipeline.model = ModelDB(
                id="977eeb18-eaac-449d-bc80-e340fbe052ad",
                name=model_name,
                format=ModelFormat.OPENVINO,
            )
            pipeline.is_running = True
        db.add(project)
        db.commit()
    click.echo("✓ Seeding successful!")


@cli.command()
def clean_db() -> None:
    """Remove all data from the database (clean but don't drop tables)."""
    with get_db_session() as db:
        db.query(DatasetItemDB).delete()
        db.query(ProjectDB).delete()
        db.query(ModelDB).delete()
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
