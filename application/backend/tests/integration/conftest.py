# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from multiprocessing.synchronize import Condition
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.models.task_type import TaskType
from app.db.schema import Base, LabelDB, ModelRevisionDB, ProjectDB, SinkDB, SourceDB
from app.schemas import OutputFormat, SinkType, SourceType
from app.schemas.model import TrainingStatus
from app.services import MetricsService
from app.services.event.event_bus import EventBus


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
def fxt_db_models() -> list[ModelRevisionDB]:
    """Fixture to create multiple models in the database."""
    return [
        ModelRevisionDB(
            training_status=TrainingStatus.NOT_STARTED,
            architecture="Object_Detection_YOLOv5",
            training_configuration={},
            label_schema_revision={},
        ),
        ModelRevisionDB(
            training_status=TrainingStatus.NOT_STARTED,
            architecture="Object_Detection_YOLOX",
            training_configuration={},
            label_schema_revision={},
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
            "id": str(uuid4()),
            "name": "Test Detection Project",
            "task_type": TaskType.DETECTION,
            "exclusive_labels": False,
        },
        {
            "id": str(uuid4()),
            "name": "Test Classification Project",
            "task_type": TaskType.CLASSIFICATION,
            "exclusive_labels": True,
        },
        {
            "id": str(uuid4()),
            "name": "Test Instance Segmentation Project",
            "task_type": TaskType.INSTANCE_SEGMENTATION,
            "exclusive_labels": True,
        },
    ]
    return [ProjectDB(**config) for config in configs]


@pytest.fixture
def fxt_db_labels() -> list[LabelDB]:
    """Fixture to create multiple labels in the database."""
    return [
        LabelDB(name="cat", color="#00FF00", hotkey="c"),
        LabelDB(name="dog", color="#FF0000", hotkey="d"),
    ]


@pytest.fixture
def fxt_event_bus() -> MagicMock:
    return MagicMock(spec=EventBus)


@pytest.fixture
def fxt_metrics_service() -> MagicMock:
    return MagicMock(spec=MetricsService)


@pytest.fixture
def fxt_condition() -> MagicMock:
    return MagicMock(spec=Condition)
