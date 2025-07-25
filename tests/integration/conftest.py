import os
import tempfile
from unittest.mock import patch

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(scope="session")
def temp_sqlite_db():
    """Create a temporary SQLite database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_path = tmp_file.name

    yield f"sqlite:///{db_path}"

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture(scope="session")
def alembic_config(temp_sqlite_db):
    """Configure Alembic for testing with temporary database."""
    alembic_cfg = Config()
    alembic_cfg.set_main_option("script_location", "app/alembic")
    alembic_cfg.set_main_option("sqlalchemy.url", temp_sqlite_db)

    return alembic_cfg


@pytest.fixture(scope="session")
def migrated_db_engine(temp_sqlite_db, alembic_config):
    """Create database engine and run migrations."""
    engine = create_engine(temp_sqlite_db, echo=False)

    # Run Alembic migrations
    command.upgrade(alembic_config, "head")

    yield engine

    engine.dispose()


@pytest.fixture(scope="function")
def db_session(migrated_db_engine):
    """Create a database session for each test."""
    SessionLocal = sessionmaker(bind=migrated_db_engine)
    session = SessionLocal()

    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(scope="function")
def mock_get_db_session(db_session):
    """Mock the get_db_session to use test database."""
    with patch("app.services.configuration_service.get_db_session") as mock:
        mock.return_value.__enter__.return_value = db_session
        mock.return_value.__exit__.return_value = None
        yield mock
