from sqlalchemy import text


def test_database_migration_applied(db_session):
    """Test that migrations have been applied successfully."""
    # Check if your tables exist
    result = db_session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
    tables = [row[0] for row in result.fetchall()]

    assert "alembic_version" in tables
    assert "sinks" in tables
    assert "pipelines" in tables
    assert "sources" in tables
