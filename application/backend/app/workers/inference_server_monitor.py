# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
from collections.abc import Callable
from functools import wraps
from multiprocessing.synchronize import Event
from uuid import UUID

from loguru import logger

from app.models.system import DeviceInfo
from app.services.inference import InferenceServer
from app.workers.base import BaseThreadWorker


class InferenceServerMonitorThread(BaseThreadWorker):
    """
    Inference Server Monitor Thread manages the lifecycle of the inference model loaded in the Inference Server,
    ensuring that models are unloaded after their TTL expires.
    It monitors the inference server for model loading, inference requests, and model stopping events to track the TTL
    countdown and unload models when necessary.
    The thread runs in the background, periodically checking the status of the loaded model and performing actions
    based on the events it observes.
    This helps to optimize resource usage by ensuring that models are not kept loaded indefinitely when they are not
    being used for inference.
    """

    ROLE = "InferenceServerMonitor"

    def __init__(self, server: InferenceServer, stop_event: Event) -> None:
        super().__init__(stop_event=stop_event)

        self._server = server
        self._ttl = 0
        self._ttl_start_time = -1.0

        self._orig_stop: Callable[[], None] | None = None

    def setup(self) -> None:
        logger.debug("Setting up inference server")

        orig_set_inference_model = self._server.set_inference_model

        @wraps(orig_set_inference_model)
        def wrapped_set_inference_model(
            project_id: UUID,
            model_id: UUID,
            device: DeviceInfo,
            ttl: int,
            model_variant_id: UUID | None = None,
        ):
            model_loaded = orig_set_inference_model(
                project_id=project_id,
                model_id=model_id,
                device=device,
                ttl=ttl,
                model_variant_id=model_variant_id,
            )
            if model_loaded:
                self._ttl = ttl
                logger.debug("Model loaded with TTL of {} seconds, starting countdown", self._ttl)
                self._ttl_start_time = time.perf_counter()
            return model_loaded

        self._server.set_inference_model = wrapped_set_inference_model

        orig_infer_batch = self._server.infer_batch

        @wraps(orig_infer_batch)
        def wrapped_infer_batch(*args, **kwargs):
            logger.debug("Batch inference requested, resetting TTL countdown")
            self._ttl_start_time = time.perf_counter()
            return orig_infer_batch(*args, **kwargs)

        self._server.infer_batch = wrapped_infer_batch

        orig_stop = self._server.stop
        self._orig_stop = orig_stop

        @wraps(orig_stop)
        def wrapped_stop():
            logger.debug("Model stopped, stopping TTL countdown")
            self._ttl_start_time = -1.0
            orig_stop()

        self._server.stop = wrapped_stop

    def run_loop(self) -> None:
        while not self.should_stop():
            if self._ttl_start_time > 0:
                elapsed = time.perf_counter() - self._ttl_start_time
                if 0 < self._ttl <= elapsed:
                    logger.debug("TTL of {} seconds expired, unloading model", self._ttl)
                    self._ttl_start_time = -1.0
                    self._orig_stop()  # pyrefly: ignore[not-callable]

            self.stop_aware_sleep(1)
