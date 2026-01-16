# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())


class BaseID(Base):
    __abstract__ = True
    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: str(uuid4()))


class SourceDB(BaseID):
    __tablename__ = "sources"

    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    config_data: Mapped[dict] = mapped_column(JSON, nullable=False)


class ProjectDB(BaseID):
    __tablename__ = "projects"
    __table_args__ = (Index("idx_projects_name", "name"),)

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)
    exclusive_labels: Mapped[bool] = mapped_column(Boolean, default=False)

    model_revisions = relationship("ModelRevisionDB", back_populates="project")


class PipelineDB(Base):
    __tablename__ = "pipelines"
    __table_args__ = (Index("idx_pipelines_is_running", "is_running"),)

    project_id: Mapped[str] = mapped_column(Text, ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True)
    source_id: Mapped[str | None] = mapped_column(Text, ForeignKey("sources.id", ondelete="RESTRICT"))
    sink_id: Mapped[str | None] = mapped_column(Text, ForeignKey("sinks.id", ondelete="RESTRICT"))
    model_revision_id: Mapped[str | None] = mapped_column(Text, ForeignKey("model_revisions.id", ondelete="RESTRICT"))
    is_running: Mapped[bool] = mapped_column(Boolean, default=False)
    data_collection: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    device: Mapped[str] = mapped_column(String(50), nullable=False, default="cpu")

    sink = relationship("SinkDB", uselist=False, lazy="joined")
    source = relationship("SourceDB", uselist=False, lazy="joined")
    model_revision = relationship("ModelRevisionDB", uselist=False, lazy="joined")


class SinkDB(BaseID):
    __tablename__ = "sinks"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    sink_type: Mapped[str] = mapped_column(String(50), nullable=False)
    rate_limit: Mapped[float | None] = mapped_column(Float, nullable=True)
    config_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    output_formats: Mapped[list] = mapped_column(JSON, nullable=False, default=list)


class ModelRevisionDB(BaseID):
    __tablename__ = "model_revisions"
    __table_args__ = (
        Index("idx_model_revisions_project_status", "project_id", "training_status"),
        Index("idx_model_revisions_architecture", "project_id", "architecture"),
    )

    project_id: Mapped[str] = mapped_column(Text, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    architecture: Mapped[str] = mapped_column(String(100), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    parent_revision: Mapped[str | None] = mapped_column(Text, ForeignKey("model_revisions.id"), nullable=True)
    training_status: Mapped[str] = mapped_column(String(50), nullable=False)
    training_configuration: Mapped[dict] = mapped_column(JSON, nullable=False)
    training_dataset_id: Mapped[str | None] = mapped_column(Text, ForeignKey("dataset_revisions.id"), nullable=True)
    label_schema_revision: Mapped[dict] = mapped_column(JSON, nullable=False)
    training_started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    training_finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    files_deleted: Mapped[bool] = mapped_column(Boolean, default=False)

    project = relationship("ProjectDB", back_populates="model_revisions")


class DatasetRevisionDB(BaseID):
    __tablename__ = "dataset_revisions"
    __table_args__ = (Index("idx_dataset_revisions_project", "project_id"),)

    project_id: Mapped[str] = mapped_column(Text, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    files_deleted: Mapped[bool] = mapped_column(Boolean, default=False)


class DatasetItemDB(BaseID):
    __tablename__ = "dataset_items"
    __table_args__ = (Index("idx_dataset_items_user_reviewed", "project_id", "user_reviewed"),)

    project_id: Mapped[str] = mapped_column(Text, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    format: Mapped[str] = mapped_column(String(50), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    size: Mapped[int] = mapped_column(Integer, nullable=False)
    annotation_data: Mapped[list | None] = mapped_column(JSON, nullable=True, default=None)
    user_reviewed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    prediction_model_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("model_revisions.id", ondelete="SET NULL"), nullable=True
    )
    source_id: Mapped[str | None] = mapped_column(Text, ForeignKey("sources.id", ondelete="SET NULL"), nullable=True)
    subset: Mapped[str | None] = mapped_column(String(20), nullable=False)
    subset_assigned_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class LabelDB(BaseID):
    __tablename__ = "labels"
    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_project_label_name"),
        UniqueConstraint("project_id", "hotkey", name="uq_project_label_hotkey"),
    )

    project_id: Mapped[str] = mapped_column(Text, ForeignKey("projects.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    color: Mapped[str] = mapped_column(String(7), nullable=False)
    hotkey: Mapped[str | None] = mapped_column(String(10), nullable=True)


class DatasetItemLabelDB(Base):
    __tablename__ = "dataset_items_labels"

    dataset_item_id: Mapped[str] = mapped_column(
        Text, ForeignKey("dataset_items.id", ondelete="CASCADE"), primary_key=True
    )
    label_id: Mapped[str] = mapped_column(Text, ForeignKey("labels.id", ondelete="CASCADE"), primary_key=True)


class TrainingConfigurationDB(BaseID):
    __tablename__ = "training_configurations"
    __table_args__ = (UniqueConstraint("project_id", "model_architecture_id", name="uq_project_model_config"),)

    project_id: Mapped[str] = mapped_column(Text, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    model_architecture_id: Mapped[str | None] = mapped_column(String(255), nullable=True)  # NULL for general config
    configuration_data: Mapped[dict] = mapped_column(JSON, nullable=False)

    project = relationship("ProjectDB")
