# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from multiprocessing.synchronize import Condition
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.schema import Base, LabelDB, ModelRevisionDB, ModelVariantDB, ProjectDB, SinkDB, SourceDB
from app.models import OutputFormat, SinkType, SourceType, TaskType, TrainingStatus
from app.models.model_revision import ModelFormat, ModelPrecision
from app.services import MetricsService, ResourceType
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
            id=str(uuid4()),
            name="YOLOX-S (abc123)",
            training_status=TrainingStatus.SUCCESSFUL,
            architecture="object-detection-yolox-s",
            training_configuration={},
            label_schema_revision={},
        ),
        ModelRevisionDB(
            id=str(uuid4()),
            name="YOLOX-X (def456)",
            training_status=TrainingStatus.SUCCESSFUL,
            architecture="object-detection-yolox-x",
            training_configuration={},
            label_schema_revision={},
        ),
    ]


@pytest.fixture
def fxt_db_model_variants(fxt_db_models) -> list[ModelVariantDB]:
    """Fixture to create FP16 OpenVINO model variants for each model revision."""
    return [
        ModelVariantDB(
            id=str(uuid4()),
            model_revision_id=model.id,
            format=ModelFormat.OPENVINO,
            precision=ModelPrecision.FP16,
        )
        for model in fxt_db_models
    ]


@pytest.fixture
def fxt_db_sources() -> list[SourceDB]:
    """Fixture to create multiple source configurations in the database."""
    return [
        SourceDB(
            id=str(uuid4()),
            source_type=SourceType.VIDEO_FILE,
            name="Test Video Source",
            config_data={"video_path": "/path/to/video.mp4"},
        ),
        SourceDB(
            id=str(uuid4()),
            source_type=SourceType.USB_CAMERA,
            name="Test USB Camera Source",
            config_data={
                "device_id": 1,
            },
        ),
        SourceDB(
            id=str(uuid4()),
            source_type=SourceType.IP_CAMERA,
            name="Test IPCamera Source",
            config_data={
                "stream_url": "rtsp://192.168.1.100:554/stream",
                "auth_required": False,
            },
        ),
    ]


@pytest.fixture
def fxt_db_sinks(tmp_path) -> list[SinkDB]:
    """Fixture to create multiple sink configurations in the database."""
    return [
        SinkDB(
            id=str(uuid4()),
            sink_type=SinkType.FOLDER,
            name="Test Folder Sink",
            rate_limit=0.2,
            output_formats=[
                OutputFormat.IMAGE_ORIGINAL,
                OutputFormat.IMAGE_WITH_PREDICTIONS,
                OutputFormat.PREDICTIONS,
            ],
            config_data={"folder_path": str(tmp_path / "test" / "path")},
        ),
        SinkDB(
            id=str(uuid4()),
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
        LabelDB(id=str(uuid4()), name="cat", color="#00FF00", hotkey="c"),
        LabelDB(id=str(uuid4()), name="dog", color="#FF0000", hotkey="d"),
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


@pytest.fixture
def fxt_entity_id(fxt_db_projects, fxt_db_models) -> Callable[[ResourceType, int], UUID]:
    """Fixture to get entity IDs by resource type and index."""

    resource_type_to_db_model = {
        ResourceType.PROJECT: fxt_db_projects,
        ResourceType.MODEL: fxt_db_models,
    }

    def get_entity_id(resource: ResourceType, idx: int) -> UUID:
        entities = resource_type_to_db_model.get(resource, [])
        if 0 <= idx < len(entities):
            return UUID(entities[idx].id)
        raise IndexError(f"{resource.value} index out of range")

    return get_entity_id


@pytest.fixture
def fxt_project_id(fxt_entity_id) -> UUID:
    """Fixture to get the first project ID."""
    return fxt_entity_id(ResourceType.PROJECT, 0)


@pytest.fixture
def fxt_model_id(fxt_entity_id) -> UUID:
    """Fixture to get the first model ID."""
    return fxt_entity_id(ResourceType.MODEL, 0)
