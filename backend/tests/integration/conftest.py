# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from multiprocessing.synchronize import Condition
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.schema import Base, ModelDB, PipelineDB, ProjectDB, SinkDB, SourceDB
from app.schemas import ModelFormat, OutputFormat, SinkType, SourceType
from app.services import ActivePipelineService, MetricsService


@pytest.fixture(scope="session")
def db_engine():
    """Create database engine and run migrations."""
    db_url = "sqlite://"
    engine = create_engine(db_url, echo=True, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """Create a database session with transaction rollback for each test."""
    connection = db_engine.connect()
    transaction = connection.begin()
    SessionLocal = sessionmaker(bind=connection)
    session = SessionLocal()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def fxt_db_models() -> list[ModelDB]:
    """Fixture to create multiple models in the database."""
    return [
        ModelDB(
            name="Test OpenVino Model",
            format=ModelFormat.OPENVINO,
        ),
        ModelDB(
            name="Test ONNX Model",
            format=ModelFormat.ONNX,
        ),
    ]


@pytest.fixture
def fxt_db_sources() -> list[SourceDB]:
    """Fixture to create multiple source configurations in the database."""
    return [
        SourceDB(
            source_type=SourceType.VIDEO_FILE,
            name="Test Video Source",
            config_data={"video_path": "/path/to/video.mp4"},
        ),
        SourceDB(
            source_type=SourceType.WEBCAM,
            name="Test Webcam Source",
            config_data={
                "device_id": 1,
            },
        ),
        SourceDB(
            source_type=SourceType.IP_CAMERA,
            name="Test IPCamera Source",
            config_data={
                "stream_url": "rtsp://192.168.1.100:554/stream",
                "auth_required": False,
            },
        ),
    ]


@pytest.fixture
def fxt_db_sinks() -> list[SinkDB]:
    """Fixture to create multiple sink configurations in the database."""
    return [
        SinkDB(
            sink_type=SinkType.FOLDER,
            name="Test Folder Sink",
            rate_limit=0.2,
            output_formats=[
                OutputFormat.IMAGE_ORIGINAL,
                OutputFormat.IMAGE_WITH_PREDICTIONS,
                OutputFormat.PREDICTIONS,
            ],
            config_data={"folder_path": "/test/path"},
        ),
        SinkDB(
            sink_type=SinkType.MQTT,
            name="Test Mqtt Sink",
            rate_limit=0.2,
            output_formats=[
                OutputFormat.IMAGE_ORIGINAL,
                OutputFormat.IMAGE_WITH_PREDICTIONS,
                OutputFormat.PREDICTIONS,
            ],
            config_data={"broker_host": "localhost", "broker_port": 1883, "topic": "topic"},
        ),
    ]


@pytest.fixture
def fxt_db_projects() -> list[ProjectDB]:
    """Fixture to create multiple projects in the database."""
    configs = [
        {
            "name": "Test Detection Project",
            "task_type": "detection",
            "exclusive_labels": False,
            "labels": ["cat", "dog"],
        },
        {
            "name": "Test Classification Project",
            "task_type": "classification",
            "exclusive_labels": True,
            "labels": ["car", "truck", "bus"],
        },
        {
            "name": "Test Segmentation Project",
            "task_type": "segmentation",
            "exclusive_labels": False,
            "labels": ["person", "bicycle"],
        },
    ]
    db_projects = []
    for config in configs:
        project = ProjectDB(**config)
        project.pipeline = PipelineDB(
            project_id=project.id,
        )
        db_projects.append(project)
    return db_projects


@pytest.fixture
def fxt_active_pipeline_service() -> MagicMock:
    return MagicMock(spec=ActivePipelineService)


@pytest.fixture
def fxt_metrics_service() -> MagicMock:
    return MagicMock(spec=MetricsService)


@pytest.fixture
def fxt_condition() -> MagicMock:
    return MagicMock(spec=Condition)
