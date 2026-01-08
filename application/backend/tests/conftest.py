# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from uuid import uuid4

import pytest

from app.core.jobs.models import Job, JobParams, JobType
from app.models import (
    ModelRevision,
    MqttSinkConfig,
    OutputFormat,
    Pipeline,
    PipelineStatus,
    SinkType,
    SourceType,
    TrainingInfo,
    TrainingStatus,
    USBCameraSourceConfig,
)
from app.models.sink import MqttConfig
from app.models.source import USBCameraConfig


@pytest.fixture
def fxt_usb_camera_source() -> USBCameraSourceConfig:
    """Sample source configuration data."""
    return USBCameraSourceConfig(
        id=uuid4(),
        source_type=SourceType.USB_CAMERA,
        name="Test Source",
        config_data=USBCameraConfig(device_id=1, codec=None),
    )


@pytest.fixture
def fxt_mqtt_sink() -> MqttSinkConfig:
    """Sample sink configuration data."""
    return MqttSinkConfig(
        id=uuid4(),
        sink_type=SinkType.MQTT,
        name="Test Sink",
        rate_limit=0.1,
        output_formats=[OutputFormat.IMAGE_WITH_PREDICTIONS],
        config_data=MqttConfig(
            broker_host="localhost",
            broker_port=1883,
            topic="topic",
            auth_required=False,
        ),
    )


@pytest.fixture
def fxt_model() -> ModelRevision:
    """Sample model data."""
    return ModelRevision(
        id=uuid4(),
        architecture="Object_Detection_YOLOX",
        training_info=TrainingInfo(status=TrainingStatus.NOT_STARTED, label_schema_revision={}, configuration={}),  # type: ignore
    )  # type: ignore


@pytest.fixture
def fxt_default_pipeline() -> Pipeline:
    """Sample default pipeline data."""
    return Pipeline(
        project_id=uuid4(),
        source=None,
        sink=None,
        model_revision=None,
        status=PipelineStatus.IDLE,
    )


@pytest.fixture
def fxt_running_pipeline(fxt_usb_camera_source, fxt_mqtt_sink, fxt_model) -> Pipeline:
    """Sample default pipeline data."""
    return Pipeline(
        project_id=uuid4(),
        source_id=fxt_usb_camera_source.id,
        sink_id=fxt_mqtt_sink.id,
        model_id=fxt_model.id,
        status=PipelineStatus.RUNNING,
    )


@pytest.fixture
def fxt_job(job_type: JobType = JobType.TRAIN, params: JobParams | None = None) -> Callable[[], Job]:
    def _factory():
        return Job(job_type=job_type, params=JobParams() if params is None else params)

    return _factory
