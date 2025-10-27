# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Application lifecycle management"""

import logging
import multiprocessing as mp
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import partial
from multiprocessing.synchronize import Condition
from pathlib import Path

from fastapi import FastAPI

from app.core.jobs import JobController, JobQueue, ProcessRunnerFactory
from app.core.run import Runnable, RunnableFactory
from app.db import MigrationManager
from app.scheduler import Scheduler
from app.schemas.job import JobType
from app.services.base_weights_service import BaseWeightsService
from app.services.data_collect import DataCollector
from app.services.event.event_bus import EventBus
from app.services.training import OTXTrainer
from app.settings import get_settings
from app.webrtc.manager import WebRTCManager

logger = logging.getLogger(__name__)


def setup_job_controller(data_dir: Path, max_parallel_jobs: int) -> tuple[JobQueue, JobController]:
    """
    Set up job controller with queue and processing infrastructure.

    Creates a job queue and controller with configured parallel job limits and training infrastructure
    for job execution.

    Args:
        data_dir: Path to the data directory.
        max_parallel_jobs (int): Maximum number of jobs that can run concurrently.

    Returns:
        tuple[JobQueue, JobController]: A tuple containing the job queue instance and the configured job controller.
    """
    q = JobQueue()
    job_runnable_factory = RunnableFactory[JobType, Runnable]()
    base_weights_service = BaseWeightsService(data_dir=data_dir)
    job_runnable_factory.register(JobType.TRAIN, partial(OTXTrainer, base_weights_service=base_weights_service))
    process_runner_factory = ProcessRunnerFactory(data_dir, job_runnable_factory)
    job_controller = JobController(
        jobs_queue=q, runner_factory=process_runner_factory, max_parallel_jobs=max_parallel_jobs
    )
    return q, job_controller


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """FastAPI lifespan context manager"""
    # Startup
    settings = get_settings()
    app.state.settings = settings
    logger.info("Starting %s application...", settings.app_name)

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
    logger.info("Shutting down %s application...", settings.app_name)
    await webrtc_manager.cleanup()
    app_scheduler.shutdown()
    logger.info("Application shutdown completed")
