# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from loguru import logger

from app.core.run import ExecutionContext, Runnable

T = TypeVar("T")


def step(name: str, complete: float | None = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to mark a method as a step of the Executor implementation.

    It expects the decorated method to be part of an Executor subclass. The decorator adds progress reporting around
    the execution of the step.

    Usage:
        class MyTrainer(Executor):
            @step("Prepare Weights")
            def prepare_weights(self, ...) -> None:
                # implementation
                pass

    Args:
        name: Human-readable name for the step (used in logging and progress reporting).
        complete: Optional float indicating the completion percentage after this step.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
            self.report_progress(f"Started: {name}")
            try:
                result = func(self, *args, **kwargs)
            except Exception:
                self.report_progress(f"Failed: {name}", exc=True)
                raise
            self.report_progress(f"Completed: {name}", percent=complete)
            return result

        return wrapper

    return decorator


class Executor(Runnable, ABC):
    """
    Abstract base class for Runnable implementations.

    Subclasses should implement their logic by defining methods decorated with @step.
    """

    def __init__(self) -> None:
        self._ctx: ExecutionContext | None = None

    @abstractmethod
    def run(self, ctx: ExecutionContext) -> None: ...

    def report_progress(self, msg: str = "", percent: float = 0.0, exc: bool = False) -> None:
        if self._ctx is not None:
            if exc:
                logger.exception(msg)
            else:
                logger.info(msg)
            self._ctx.report(msg, percent)
