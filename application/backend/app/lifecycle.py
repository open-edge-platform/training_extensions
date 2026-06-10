# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Application lifecycle management"""

import multiprocessing as mp
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import partial
from multiprocessing.synchronize import Condition
from pathlib import Path

from aiortc import RTCConfiguration, RTCIceServer
from fastapi import FastAPI
from loguru import logger

from app.core.jobs import JobController, JobQueue, ProcessRunnerFactory
from app.core.jobs.models import JobType
from app.core.logging import LogConfig, setup_logging, setup_uvicorn_logging
from app.core.run import Runnable, RunnableFactory
from app.db import MigrationManager, get_db_session
from app.execution.builders import (
    build_export_dataset,
    build_import_as_new_project,
    build_import_to_project,
    build_prepare_dataset,
    build_quantizer,
    build_trainer,
)
from app.scheduler import Scheduler
from app.services import (
    DatasetRevisionService,
    DatasetService,
    LabelService,
    MediaService,
    ModelService,
    PipelineService,
    ProjectService,
    TrainingConfigurationService,
)
from app.services.base_weights_service import BaseWeightsService
from app.services.data_collect import DataCollector
from app.services.event.event_bus import EventBus
from app.services.inference import InferenceServer
from app.services.subset_assignment import SubsetAssigner, SubsetService
from app.services.video import CacheConfig, VideoService
from app.settings import get_settings
from app.webrtc import SDPHandler, WebRTCManager, WebRTCSettings


def setup_job_controller(
    data_dir: Path, staged_datasets_dir: Path | None, max_parallel_jobs: int
) -> tuple[JobQueue, JobController]:
    """
    Initializes and configures the job queue and job controller for managing parallel job execution.

    Sets up the infrastructure to run jobs concurrently and registers classes that comply with the Runnable protocol,
    each associated with a job type and its required dependencies. These classes are executed in a context defined
    by the runner factory.

    Args:
        data_dir: Path to the directory containing data required for job execution.
        staged_datasets_dir: Path to the directory for storing staged datasets.
        max_parallel_jobs (int): Maximum number of jobs that can run concurrently.

    Returns:
        tuple[JobQueue, JobController]: The job queue and the configured job controller.
    """
    if not staged_datasets_dir:
        raise ValueError("staged_datasets_dir must be provided")
    q = JobQueue()
    job_runnable_factory = RunnableFactory[JobType, Runnable]()
    label_service = LabelService()
    dataset_service = DatasetService(
        label_service=label_service,
        media_service=MediaService(data_dir=data_dir),
    )
    project_service = ProjectService(
        data_dir=data_dir,
        label_service=label_service,
        pipeline_service=PipelineService(),
    )
    dataset_revision_service = DatasetRevisionService(data_dir=data_dir)
    job_runnable_factory.register(
        JobType.TRAIN,
        partial(
            build_trainer,
            base_weights_service=BaseWeightsService(data_dir=data_dir),
            subset_service=SubsetService(),
            subset_assigner=SubsetAssigner(),
            dataset_service=dataset_service,
            dataset_revision_service=dataset_revision_service,
            model_service=ModelService(data_dir=data_dir),
            training_configuration_service=TrainingConfigurationService(),
            data_dir=data_dir,
            db_session_factory=get_db_session,
        ),
    )
    job_runnable_factory.register(
        JobType.QUANTIZE,
        partial(
            build_quantizer,
            data_dir=data_dir,
            model_service=ModelService(data_dir=data_dir),
            dataset_revision_service=dataset_revision_service,
            project_service=project_service,
            training_configuration_service=TrainingConfigurationService(),
            db_session_factory=get_db_session,
        ),
    )
    job_runnable_factory.register(
        JobType.EXPORT_DATASET,
        partial(
            build_export_dataset,
            staged_datasets_dir=staged_datasets_dir,
            dataset_service=dataset_service,
            dataset_revision_service=dataset_revision_service,
            db_session_factory=get_db_session,
        ),
    )
    job_runnable_factory.register(
        JobType.PREPARE_DATASET_FOR_IMPORT,
        partial(
            build_prepare_dataset,
            staged_datasets_dir=staged_datasets_dir,
        ),
    )
    job_runnable_factory.register(
        JobType.IMPORT_DATASET_TO_PROJECT,
        partial(
            build_import_to_project,
            staged_datasets_dir=staged_datasets_dir,
            dataset_service=dataset_service,
            label_service=label_service,
            media_service=MediaService(data_dir=data_dir),
            db_session_factory=get_db_session,
        ),
    )
    job_runnable_factory.register(
        JobType.IMPORT_DATASET_AS_NEW_PROJECT,
        partial(
            build_import_as_new_project,
            staged_datasets_dir=staged_datasets_dir,
            project_service=project_service,
            dataset_service=dataset_service,
            label_service=label_service,
            media_service=MediaService(data_dir=data_dir),
            db_session_factory=get_db_session,
        ),
    )
    process_runner_factory = ProcessRunnerFactory(job_runnable_factory)
    job_controller = JobController(
        jobs_queue=q, runner_factory=process_runner_factory, max_parallel_jobs=max_parallel_jobs
    )
    return q, job_controller


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """FastAPI lifespan context manager"""
    # Startup
    settings = get_settings()
    settings.ensure_dirs_exist()
    app.state.settings = settings
    logger.info("Starting {} application...", settings.app_name)

    # Setup logging
    setup_logging(config=LogConfig(level=settings.log_level))
    setup_uvicorn_logging(settings.log_level)

    # Initialize database
    migration_manager = MigrationManager(settings)
    if not migration_manager.initialize_database():
        logger.error("Failed to initialize database. Application cannot start.")
        raise RuntimeError("Database initialization failed")

    # Worker processes are created with the "spawn" method to ensure a clean state and avoid issues with shared
    # resources, especially when the workers involve GPU usage or complex libraries that may not be fork-safe.
    # See https://github.com/open-edge-platform/training_extensions/issues/5701 for more details.
    mp_ctx = mp.get_context("spawn")

    # Condition to notify processes about source updates
    source_changed_condition: Condition = mp_ctx.Condition()
    # Event to signal that the model has to be reloaded
    model_reload_event = mp_ctx.Event()

    event_bus = EventBus(source_changed_condition=source_changed_condition, model_reload_event=model_reload_event)
    app.state.event_bus = event_bus

    cache_config = CacheConfig(
        ttl=settings.video_cache_ttl,
        cleanup_interval=settings.video_cache_cleanup_interval,
    )
    video_service = VideoService(cache_config=cache_config)
    app.state.video_service = video_service

    data_collector = DataCollector(data_dir=settings.data_dir, event_bus=event_bus)
    app.state.data_collector = data_collector

    inference_server = InferenceServer(data_dir=settings.data_dir)
    app.state.inference_server = inference_server

    # Initialize Scheduler
    app_scheduler = Scheduler(
        event_bus=event_bus, data_collector=data_collector, inference_server=inference_server, mp_ctx=mp_ctx
    )
    app_scheduler.start_workers()
    app.state.scheduler = app_scheduler

    webrtc_settings = WebRTCSettings(
        config=RTCConfiguration(iceServers=[RTCIceServer(**server) for server in settings.ice_servers]),
        advertise_ip=settings.webrtc_advertise_ip,
    )
    sdp_handler = SDPHandler()
    webrtc_manager = WebRTCManager(app_scheduler.rtc_stream_broadcaster, webrtc_settings, sdp_handler)
    app.state.webrtc_manager = webrtc_manager
    logger.info("Application startup completed")

    job_queue, job_controller = setup_job_controller(
        data_dir=settings.data_dir,
        staged_datasets_dir=settings.staged_datasets_dir,
        max_parallel_jobs=settings.gpu_slots,
    )
    app.state.job_queue = job_queue

    await job_controller.start()

    yield

    await job_controller.stop()
    # Shutdown
    logger.info("Shutting down {} application...", settings.app_name)
    video_service.close()
    await webrtc_manager.cleanup()
    app_scheduler.shutdown()
    logger.info("Application shutdown completed")
