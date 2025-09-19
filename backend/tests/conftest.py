# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest

from app.schemas import Model, ModelFormat, OutputFormat, Pipeline, PipelineStatus, SinkType, SourceType
from app.schemas.sink import MqttSinkConfig
from app.schemas.source import WebcamSourceConfig


@pytest.fixture
def fxt_webcam_source() -> WebcamSourceConfig:
    """Sample source configuration data."""
    return WebcamSourceConfig(source_type=SourceType.WEBCAM, name="Test Source", device_id=1)


@pytest.fixture
def fxt_mqtt_sink() -> MqttSinkConfig:
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
def fxt_model() -> Model:
    """Sample model data."""
    return Model(name="YOLO-X for Vehicle Detection", format=ModelFormat.OPENVINO)


@pytest.fixture
def fxt_default_pipeline() -> Pipeline:
    """Sample default pipeline data."""
    return Pipeline(
        project_id=uuid4(),
        source=None,
        sink=None,
        model=None,
        status=PipelineStatus.IDLE,
    )


@pytest.fixture
def fxt_running_pipeline(fxt_webcam_source, fxt_mqtt_sink, fxt_model) -> Pipeline:
    """Sample default pipeline data."""
    return Pipeline(
        project_id=uuid4(),
        source_id=fxt_webcam_source.id,
        sink_id=fxt_mqtt_sink.id,
        model_id=fxt_model.id,
        status=PipelineStatus.RUNNING,
    )
