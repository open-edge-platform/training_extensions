from uuid import uuid4

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class SourceDB(Base):
    __tablename__ = "sources"

    id = Column(Text, primary_key=True, default=lambda: str(uuid4()))
    source_type = Column(String(50), nullable=False)
    config_data = Column(JSON, nullable=False)
    created_at = Column(DateTime, server_default=func.current_timestamp())
    updated_at = Column(DateTime, server_default=func.current_timestamp())


class PipelineDB(Base):
    __tablename__ = "pipelines"

    id = Column(Text, primary_key=True, default=lambda: str(uuid4()))
    source_id = Column(Text, ForeignKey("sources.id", ondelete="SET NULL"))
    sink_id = Column(Text, ForeignKey("sinks.id", ondelete="SET NULL"))
    name = Column(String(255), unique=True, nullable=False)
    description = Column(Text)
    is_running = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.current_timestamp())
    updated_at = Column(DateTime, server_default=func.current_timestamp())


class SinkDB(Base):
    __tablename__ = "sinks"

    id = Column(Text, primary_key=True, default=lambda: str(uuid4()))
    destination_type = Column(String(50), nullable=False)
    rate_limit = Column(Float, nullable=True)
    config_data = Column(JSON, nullable=False)
    output_formats = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime, server_default=func.current_timestamp())
    updated_at = Column(DateTime, server_default=func.current_timestamp())
