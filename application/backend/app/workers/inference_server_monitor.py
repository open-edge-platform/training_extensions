# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
from collections.abc import Callable
from functools import wraps
from multiprocessing.synchronize import Event

from loguru import logger

from app.services.inference import InferenceServer
from app.workers.base import BaseThreadWorker


class InferenceServerMonitorThread(BaseThreadWorker):
    ROLE = "InferenceServer"

    def __init__(self, server: InferenceServer, stop_event: Event) -> None:
        super().__init__(stop_event=stop_event)

        self._server = server
        self._model_loaded = False
        self._model_stopped = False
        self._inference_requested = False
        self._ttl = 0
        self._ttl_start_time = -1.0

        self._orig_stop: Callable[[], None] | None = None

    def setup(self) -> None:
        logger.debug("Setting up inference server")

        orig_set_inference_model = self._server.set_inference_model

        @wraps(orig_set_inference_model)
        def wrapped_set_inference_model(*args, **kwargs):
            model_loaded = orig_set_inference_model(*args, **kwargs)
            if model_loaded:
                self._ttl = kwargs.get("ttl")  # pyrefly: ignore[bad-assignment]
                logger.debug("Model loaded with TTL of {} seconds, starting countdown", self._ttl)
                self._model_loaded = True
            return model_loaded

        self._server.set_inference_model = wrapped_set_inference_model

        orig_infer_batch = self._server.infer_batch

        @wraps(orig_infer_batch)
        def wrapped_infer_batch(*args, **kwargs):
            self._inference_requested = True
            return orig_infer_batch(*args, **kwargs)

        self._server.infer_batch = wrapped_infer_batch

        orig_stop = self._server.stop
        self._orig_stop = orig_stop

        @wraps(orig_stop)
        def wrapped_stop():
            self._model_stopped = True
            orig_stop()

        self._server.stop = wrapped_stop

    def run_loop(self) -> None:
        while not self.should_stop():
            if self._model_loaded:
                logger.debug("Model loaded with TTL of {} seconds, starting countdown", self._ttl)
                self._model_loaded = False
                self._ttl_start_time = time.perf_counter()

            if self._model_stopped:
                logger.debug("Model stopped, stopping TTL countdown")
                self._model_stopped = False
                self._ttl_start_time = -1.0

            if self._inference_requested:
                logger.debug("Batch inference requested, resetting TTL countdown")
                self._inference_requested = False
                self._ttl_start_time = time.perf_counter()

            if self._ttl_start_time > 0:
                elapsed = time.perf_counter() - self._ttl_start_time
                if 0 < self._ttl <= elapsed:
                    logger.debug("TTL of {} seconds expired, unloading model", self._ttl)
                    self._ttl_start_time = -1.0
                    self._orig_stop()  # pyrefly: ignore[not-callable]
