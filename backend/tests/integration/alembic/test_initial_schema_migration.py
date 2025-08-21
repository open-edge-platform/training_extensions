# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tempfile

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture(scope="session")
def migrated_db_engine():
    """Create database engine and run migrations."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp_file:
        db_url = f"sqlite:///{tmp_file.name}"
        engine = create_engine(db_url, echo=True, poolclass=StaticPool)

        alembic_cfg = Config()
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)
        alembic_cfg.set_main_option("script_location", "app/alembic")

        command.upgrade(alembic_cfg, "head")

        yield engine

        engine.dispose()


@pytest.fixture
def alembic_session(migrated_db_engine):
    """Create a database session with transaction rollback for each test."""
    SessionLocal = sessionmaker(bind=migrated_db_engine)
    session = SessionLocal()

    try:
        yield session
    finally:
        session.close()


def test_database_migration_applied(alembic_session):
    """Test that migrations have been applied successfully."""
    # Check if your tables exist
    result = alembic_session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
    tables = [row[0] for row in result.fetchall()]

    assert len(tables) == 5
    assert "alembic_version" in tables
    assert "sinks" in tables
    assert "pipelines" in tables
    assert "sources" in tables
    assert "models" in tables

    (result,) = alembic_session.execute(text("SELECT version_num FROM alembic_version")).fetchone()
    assert result == "cb68fa7db781"
