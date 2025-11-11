# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from loguru import logger

from app.core.run import ExecutionContext, Runnable

from .models import TrainingParams

T = TypeVar("T")


def step(name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to mark a method as a training step.

    It expects the decorated method to be part of a Trainer subclass. The decorator adds progress reporting around the
    execution of the step.

    Usage:
        class MyTrainer(Trainer):
            @step("Prepare Weights")
            def prepare_weights(self, ...) -> None:
                # implementation
                pass

    Args:
        name: Human-readable name for the step (used in logging and progress reporting).
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
            self.report_progress(f"Starting: {name}")
            try:
                result = func(self, *args, **kwargs)
            except Exception:
                self.report_progress(f"Failed: {name}")
                raise
            self.report_progress(f"Completed: {name}")
            return result

        return wrapper

    return decorator


class Trainer(Runnable, ABC):
    """
    Abstract base class for model training workflows.

    Subclasses should implement their training logic by defining methods decorated with @step.
    """

    def __init__(self) -> None:
        self._ctx: ExecutionContext | None = None
        self._training_params: TrainingParams | None = None

    @abstractmethod
    def run(self, ctx: ExecutionContext) -> None: ...

    @staticmethod
    def _get_training_params(ctx: ExecutionContext) -> TrainingParams:
        return TrainingParams.model_validate_json(ctx.payload)

    def report_progress(self, msg: str = "", percent: float = 0.0) -> None:
        if self._ctx is not None:
            logger.info(msg)
            self._ctx.report(msg, percent)

    def heartbeat(self) -> None:
        if self._ctx is not None:
            self._ctx.heartbeat()
