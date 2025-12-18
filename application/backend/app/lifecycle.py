# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Application lifecycle management"""

import multiprocessing as mp
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import partial
from multiprocessing.synchronize import Condition
from pathlib import Path

from fastapi import FastAPI
from loguru import logger

from app.core.jobs import JobController, JobQueue, ProcessRunnerFactory
from app.core.logging import LogConfig, setup_logging, setup_uvicorn_logging
from app.core.run import Runnable, RunnableFactory
from app.db import MigrationManager, get_db_session
from app.scheduler import Scheduler
from app.schemas.job import JobType
from app.services import (
    DatasetRevisionService,
    DatasetService,
    LabelService,
    ModelService,
    TrainingConfigurationService,
)
from app.services.base_weights_service import BaseWeightsService
from app.services.data_collect import DataCollector
from app.services.event.event_bus import EventBus
from app.services.training import OTXTrainer
from app.services.training.otx_trainer import TrainingDependencies
from app.services.training.subset_assignment import SubsetAssigner, SubsetService
from app.settings import get_settings
from app.webrtc.manager import WebRTCManager


def setup_job_controller(data_dir: Path, max_parallel_jobs: int) -> tuple[JobQueue, JobController]:
    """
    Initializes and configures the job queue and job controller for managing parallel job execution.

    Sets up the infrastructure to run jobs concurrently and registers classes that comply with the Runnable protocol,
    each associated with a job type and its required dependencies. These classes are executed in a context defined
    by the runner factory.

    Args:
        data_dir: Path to the directory containing data required for job execution.
        max_parallel_jobs (int): Maximum number of jobs that can run concurrently.

    Returns:
        tuple[JobQueue, JobController]: The job queue and the configured job controller.
    """
    q = JobQueue()
    job_runnable_factory = RunnableFactory[JobType, Runnable]()
    job_runnable_factory.register(
        JobType.TRAIN,
        partial(
            OTXTrainer,
            training_deps=TrainingDependencies(
                base_weights_service=BaseWeightsService(data_dir=data_dir),
                subset_service=SubsetService(),
                subset_assigner=SubsetAssigner(),
                dataset_service=DatasetService(data_dir=data_dir, label_service=LabelService()),
                dataset_revision_service=DatasetRevisionService(data_dir=data_dir),
                model_service=ModelService(data_dir=data_dir),
                training_configuration_service=TrainingConfigurationService(),
                data_dir=data_dir,
                db_session_factory=get_db_session,
            ),
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

    # Condition to notify processes about source updates
    source_changed_condition: Condition = mp.Condition()

    event_bus = EventBus(source_changed_condition=source_changed_condition)
    app.state.event_bus = event_bus

    data_collector = DataCollector(data_dir=settings.data_dir, event_bus=event_bus)
    app.state.data_collector = data_collector

    # Initialize Scheduler
    app_scheduler = Scheduler(event_bus=event_bus, data_collector=data_collector)
    app_scheduler.start_workers()
    app.state.scheduler = app_scheduler

    webrtc_manager = WebRTCManager(app_scheduler.rtc_stream_queue)
    app.state.webrtc_manager = webrtc_manager
    logger.info("Application startup completed")

    job_queue, job_controller = setup_job_controller(data_dir=settings.data_dir, max_parallel_jobs=settings.gpu_slots)
    app.state.job_queue = job_queue

    await job_controller.start()

    yield

    await job_controller.stop()
    # Shutdown
    logger.info("Shutting down {} application...", settings.app_name)
    await webrtc_manager.cleanup()
    app_scheduler.shutdown()
    logger.info("Application shutdown completed")
