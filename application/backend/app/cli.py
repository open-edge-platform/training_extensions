# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Command line interface for interacting with the Geti Tune application."""

import sys
from pathlib import Path

import click

from app.db import MigrationManager
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


@cli.command()
@click.option("--project-id", required=True, help="ID of the project to export")
@click.option("--output", required=True, help="Output path for zip archive")
def export_project(project_id: str, output: str) -> None:
    """Export project data from SQLite database to a zip archive."""
    from app.db.import_export.export_project import export_project as do_export

    try:
        do_export(project_id=project_id, output_archive=output)
        click.echo("✓ Project exported successfully!")
    except Exception as e:
        click.echo(f"✗ Export failed: {e}")
        sys.exit(1)


@cli.command()
@click.option("--input", "input_archive", required=True, help="Path to zip archive containing project data")
@click.option(
    "-f",
    "--force-import",
    is_flag=True,
    help="Bypass database schema version check and attempt import anyway (use at your own risk)",
)
def import_project(input_archive: str, force_import: bool) -> None:
    """Import project data from a zip archive into the SQLite database."""
    from app.db.import_export.import_project import import_project as do_import

    try:
        do_import(input_archive=input_archive, allow_mismatching_db_schema=force_import)
        click.echo("✓ Project imported successfully!")
    except Exception as e:
        click.echo(f"✗ Import failed: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--video-path",
    "video_paths",
    multiple=True,
    required=True,
    help="Path(s) to local video files to register as sources (can be specified multiple times)",
)
def setup_demo_sources(video_paths: tuple[str, ...]) -> None:
    """Register local video files as sources in the database."""
    from app.db.engine import get_db_session
    from app.db.schema import SourceDB

    sources_to_create: list[SourceDB] = []

    for video_path_str in video_paths:
        video_path = Path(video_path_str).resolve()

        # Validate that the file exists
        if not video_path.exists():
            click.echo(f"✗ Video file not found: {video_path}")
            sys.exit(1)

        if not video_path.is_file():
            click.echo(f"✗ Path is not a file: {video_path}")
            sys.exit(1)

        click.echo(f"Registering video: {video_path.name}")

        # Convert to relative path from current working directory
        # This ensures the path includes "data/" prefix
        cwd = Path.cwd()
        data_dir_abs = settings.data_dir.resolve()

        try:
            # Check if video is under data directory
            video_path.relative_to(data_dir_abs)
            # Calculate relative path from cwd (which includes "data/")
            video_path_for_db = str(video_path.relative_to(cwd))
        except ValueError:
            # If video is not under data_dir, use absolute path as fallback
            click.echo(f"⚠ Warning: Video path is not under data directory ({settings.data_dir})")
            video_path_for_db = str(video_path)

        # Create source record
        source_name = video_path.stem.replace("-", " ").replace("_", " ").title()
        source = SourceDB(
            name=source_name,
            source_type="video_file",
            config_data={"video_path": video_path_for_db},
        )
        sources_to_create.append(source)

    # Insert sources into database
    if sources_to_create:
        click.echo(f"Creating {len(sources_to_create)} source record(s) in database...")
        try:
            with get_db_session() as db:
                db.add_all(sources_to_create)
                db.flush()
                db.commit()
            click.echo(f"✓ Created {len(sources_to_create)} demo source(s) successfully!")
        except Exception as e:
            click.echo(f"✗ Failed to create sources: {e}")
            sys.exit(1)


@cli.command()
def setup_demo_sinks() -> None:
    """Create default sinks for demo purposes."""
    from app.db.engine import get_db_session
    from app.db.schema import SinkDB

    # Define the default output folder sink
    output_path = settings.data_dir / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    sink = SinkDB(
        name="Default output folder",
        sink_type="folder",
        rate_limit=0.2,
        config_data={"folder_path": str(output_path)},
        output_formats=["image_with_predictions", "predictions"],
    )

    click.echo(f"Creating default folder sink: {sink.name}")
    try:
        with get_db_session() as db:
            db.add(sink)
            db.flush()
            db.commit()
        click.echo("✓ Created demo sink successfully!")
    except Exception as e:
        click.echo(f"✗ Failed to create sink: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
