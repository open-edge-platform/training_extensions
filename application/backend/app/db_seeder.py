# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from datetime import datetime, timedelta
from uuid import UUID

from app.db.schema import LabelDB, ModelRevisionDB, PipelineDB, ProjectDB, SinkDB, SourceDB
from app.models import (
    DisconnectedSinkConfig,
    DisconnectedSourceConfig,
    FixedRateDataCollectionPolicy,
    OutputFormat,
    SinkType,
    SourceType,
    TaskType,
    TrainingStatus,
)


def _create_shared_sinks_sources_folders() -> tuple[SourceDB, SinkDB, SinkDB]:
    """
    Create shared source, sink, folder entities.

    Returns:
        tuple[SourceDB, SinkDB, SinkDB]: Created source, sink, and folder sink objects.
    """
    disconnected_source_cfg = DisconnectedSourceConfig()
    disconnected_source = SourceDB(
        id="00000000-0000-0000-0000-000000000000",
        name=disconnected_source_cfg.name,
        source_type=disconnected_source_cfg.source_type,
        config_data={},
    )
    disconnected_sink_cfg = DisconnectedSinkConfig()
    disconnected_sink = SinkDB(
        id="00000000-0000-0000-0000-000000000000",
        name=disconnected_sink_cfg.name,
        sink_type=disconnected_sink_cfg.sink_type,
        output_formats=[],
        config_data={},
    )
    folder_sink = SinkDB(
        id="6ee0c080-c7d9-4438-a7d2-067fd395eecf",
        name="Folder Sink",
        sink_type=SinkType.FOLDER,
        rate_limit=0.2,
        output_formats=[OutputFormat.IMAGE_ORIGINAL, OutputFormat.IMAGE_WITH_PREDICTIONS, OutputFormat.PREDICTIONS],
        config_data={"folder_path": "data/output"},
    )
    return disconnected_source, disconnected_sink, folder_sink


def _create_project(
    project_id: str | UUID,
    task_type: TaskType,
    exclusive_labels: bool = True,
) -> ProjectDB:
    """
    Create a project in the database.

    Args:
        project_id (str | UUID): Unique identifier for the project.
        task_type (TaskType): Type of task (e.g., DETECTION, INSTANCE_SEGMENTATION, CLASSIFICATION).
        exclusive_labels (bool): Whether labels are mutually exclusive.

    Returns:
        ProjectDB: Created project object.
    """
    return ProjectDB(
        id=project_id,
        name=f"Demo {task_type} project",
        task_type=task_type,
        exclusive_labels=exclusive_labels,
    )


def _create_detection_labels(project_id: str | UUID) -> list[LabelDB]:
    """
    Create labels for a Detection card project.

    Args:
        project_id (str | UUID): ID of the project to add labels to.

    Returns:
        list[LabelDB]: List of created label objects.
    """
    return [
        LabelDB(project_id=project_id, name="Clubs", color="#2d6311", hotkey="c"),
        LabelDB(project_id=project_id, name="Diamonds", color="#baa3b3", hotkey="d"),
        LabelDB(project_id=project_id, name="Spades", color="#000702", hotkey="s"),
        LabelDB(project_id=project_id, name="Hearts", color="#1f016b", hotkey="h"),
        LabelDB(project_id=project_id, name="No_object", color="#565a84", hotkey="n"),
    ]


def _create_segmentation_labels(project_id: str | UUID) -> list[LabelDB]:
    """
    Create labels for an Instance Segmentation fish project.

    Args:
        project_id (str | UUID): ID of the project to add labels to.

    Returns:
        list[LabelDB]: List of created label objects.
    """
    return [
        LabelDB(project_id=project_id, name="Fish", color="#2d6311", hotkey="f"),
        LabelDB(project_id=project_id, name="Empty", color="#565a84", hotkey="e"),
    ]


def _create_pipeline_with_video_source(  # noqa: PLR0913
    project_id: str | UUID,
    source_id: str | UUID,
    source_name: str,
    video_path: str,
    sink_id: str | UUID,
    model_id: str,
    model_architecture: str,
    labels: list[LabelDB],
) -> PipelineDB:
    """
    Create a pipeline with a video file source for a project.

    Args:
        project_id (str | UUID): ID of the project.
        source_id (str | UUID): Unique identifier for the video source.
        source_name (str): Name for the video source.
        video_path (str): Path to the video file.
        sink_id (str | UUID): ID of the sink to use.
        model_id (str): Unique identifier for the model revision.
        model_architecture (str): Architecture name of the model.
        labels (list[LabelDB] | None): List of labels for the label schema revision.

    Returns:
        PipelineDB: Created pipeline object.
    """
    pipeline = PipelineDB(
        project_id=project_id,
        sink_id=sink_id,
        data_collection_policies=[FixedRateDataCollectionPolicy(rate=0.1).model_dump(mode="json")],
        is_running=project_id == "9d6af8e8-6017-4ebe-9126-33aae739c5fa",  # Running only for detection project
    )

    pipeline.source = SourceDB(
        id=source_id,
        name=source_name,
        source_type=SourceType.VIDEO_FILE,
        config_data={"video_path": video_path},
    )

    pipeline.model_revision = ModelRevisionDB(
        id=model_id,
        project_id=project_id,
        architecture=model_architecture,
        training_status=TrainingStatus.SUCCESSFUL,
        training_started_at=datetime.now() - timedelta(hours=24),
        training_finished_at=datetime.now() - timedelta(hours=23),
        training_configuration={},
        label_schema_revision={"labels": [{"id": str(label.id), "name": label.name} for label in labels]},
    )
    return pipeline
