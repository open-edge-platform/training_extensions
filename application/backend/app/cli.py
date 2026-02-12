# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Command line interface for interacting with the Geti Tune application."""

import sys

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


if __name__ == "__main__":
    cli()
