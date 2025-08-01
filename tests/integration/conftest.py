import tempfile
from collections.abc import Generator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.schema import PipelineDB


@pytest.fixture(scope="session")
def temp_sqlite_db():
    """Create a temporary SQLite database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp_file:
        db_path = tmp_file.name
        yield f"sqlite:///{db_path}"


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


@pytest.fixture
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
def default_pipeline(db_session) -> Generator[PipelineDB]:
    """Seed the database with default pipeline."""
    pipeline = PipelineDB(name="Default Pipeline", is_running=True)
    db_session.add(pipeline)
    db_session.commit()
    try:
        yield pipeline
    finally:
        obj = db_session.query(PipelineDB).filter_by(is_running=True).one()
        db_session.delete(obj)
        db_session.commit()
