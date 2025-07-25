import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import NullPool

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/geti_edge.db")
DB_ECHO = os.getenv("DB_ECHO", "0").lower() in ("1", "true", "yes")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
    # https://docs.sqlalchemy.org/en/14/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork
    poolclass=NullPool,
    echo=DB_ECHO,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


@contextmanager
def get_db_session() -> Session:
    """Context manager to get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
