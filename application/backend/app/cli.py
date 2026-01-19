# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Command line interface for interacting with the Geti Tune application."""

import sys

import click

from app.db import MigrationManager, get_db_session
from app.db.schema import DatasetItemDB, ModelRevisionDB, ProjectDB, SinkDB, SourceDB
from app.db_seeder import (
    _create_detection_labels,
    _create_pipeline_with_video_source,
    _create_project,
    _create_segmentation_labels,
    _create_shared_sinks_sources_folders,
)
from app.models import TaskType
from app.settings import get_settings

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
    """
    Seed the database with test data.

    Args:
        with_model (bool): Whether to include pre-trained models in the seed data.
    """
    # If the app is running, it needs to be restarted since it doesn't track direct DB changes
    # Fixed IDs are used to ensure consistency in tests
    click.echo("Seeding database with test data...")
    sources, sinks, folders = _create_shared_sinks_sources_folders()

    # Project 1: Object Detection
    detection_project = _create_project(
        project_id="9d6af8e8-6017-4ebe-9126-33aae739c5fa",
        task_type=TaskType.DETECTION,
        exclusive_labels=True,
    )
    detection_labels = _create_detection_labels(project_id=detection_project.id)

    # Project 2: Instance Segmentation
    segmentation_project = _create_project(
        project_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        task_type=TaskType.INSTANCE_SEGMENTATION,
        exclusive_labels=True,
    )
    segmentation_labels = _create_segmentation_labels(project_id=segmentation_project.id)

    detection_pipeline = None
    instance_segmentation_pipeline = None
    if with_model:
        detection_pipeline = _create_pipeline_with_video_source(
            project_id=detection_project.id,
            source_id="f6b1ac22-e36c-4b36-9a23-62b0881e4223",
            source_name="Video Source - Detection",
            video_path="data/media/card-video.mp4",
            sink_id=folders.id,
            model_id="977eeb18-eaac-449d-bc80-e340fbe052ad",
            model_architecture="Custom_Object_Detection_Gen3_SSD",
            labels=detection_labels,
        )

        instance_segmentation_pipeline = _create_pipeline_with_video_source(
            project_id=segmentation_project.id,
            source_id="b2c3d4e5-f6a7-8901-bcde-f12345678901",
            source_name="Video Source - Segmentation",
            video_path="data/media/fish-video.mp4",
            sink_id=folders.id,
            model_id="c3d4e5f6-a7b8-9012-cdef-123456789012",
            model_architecture="Custom_Instance_Segmentation_RTMDet_tiny",
            labels=segmentation_labels,
        )

    with get_db_session() as db:
        db.add_all([sources, sinks, folders, detection_project, segmentation_project])
        db.flush()
        db.add_all(detection_labels + segmentation_labels)
        db.flush()

        if with_model and detection_pipeline and instance_segmentation_pipeline:
            db.add_all([detection_pipeline, instance_segmentation_pipeline])
            db.flush()

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
        click.echo("OpenAPI specification generated successfully!")
    except Exception as e:
        click.echo(f"Failed to generate OpenAPI specification: {e}")
        sys.exit(1)
    click.echo("Waiting for threading to finish...")


if __name__ == "__main__":
    cli()
