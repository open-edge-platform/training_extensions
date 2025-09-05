# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator
from multiprocessing.synchronize import Condition
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.schema import Base, ModelDB, PipelineDB, SinkDB, SourceDB
from app.schemas import ModelFormat, OutputFormat, SinkType, SourceType
from app.schemas.sink import MqttSinkConfig
from app.schemas.source import WebcamSourceConfig
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
def fxt_default_pipeline(db_session) -> Generator[PipelineDB]:
    """Seed the database with default pipeline."""
    pipeline = PipelineDB(name="Default Pipeline", is_running=True)
    db_session.add(pipeline)
    yield pipeline


@pytest.fixture
def fxt_source_config() -> WebcamSourceConfig:
    """Sample source configuration data."""
    return WebcamSourceConfig(source_type=SourceType.WEBCAM, name="Test Source", device_id=1)


@pytest.fixture
def fxt_sink_config() -> MqttSinkConfig:
    """Sample sink configuration data."""
    return MqttSinkConfig(
        sink_type=SinkType.MQTT,
        name="Test Sink",
        rate_limit=0.1,
        output_formats=[OutputFormat.IMAGE_WITH_PREDICTIONS],
        broker_host="localhost",
        broker_port=1883,
        topic="topic",
    )


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
def fxt_active_pipeline_service() -> MagicMock:
    return MagicMock(spec=ActivePipelineService)


@pytest.fixture
def fxt_metrics_service() -> MagicMock:
    return MagicMock(spec=MetricsService)


@pytest.fixture
def fxt_condition() -> MagicMock:
    return MagicMock(spec=Condition)
