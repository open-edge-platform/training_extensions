# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Runnable protocol and factory definitions.

This module defines the generic interface for activities (Runnables) that can be executed by runners, along with a
factory for registering and instantiating Runnable implementations. Runnables encapsulate the logic to be executed with
a given execution context, supporting progress reporting and heartbeats.

Classes:
    ExecutionContext: Carries execution parameters, progress reporting, and heartbeat callbacks.
    Runnable: Protocol for executable activities.
    RunnableFactory: Factory for registering and creating Runnable instances by type.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Protocol, TypeVar

ReportFn = Callable[[str, float], None]
HeartbeatFn = Callable[[], None]


@dataclass(kw_only=True)
class ExecutionContext:
    payload: str
    data_dir: Path
    report: ReportFn
    heartbeat: HeartbeatFn

    def report_progress(self, msg: str = "", progress: float = 0.0) -> None:
        """Report progress of the execution."""
        self.report(msg, progress)


class Runnable(Protocol):  # ignore
    """Generic interface for activities executed by runners."""

    def run(self, ctx: ExecutionContext) -> None: ...


R = TypeVar("R", bound=Runnable)
K = TypeVar("K")


class RunnableFactory(Generic[K, R]):
    """Factory protocol for creating Runnable instances."""

    def __init__(self) -> None:
        self._registry: dict[K, Callable[[], R]] = {}

    def __call__(self, cls_type: K) -> R:
        if cls_type not in self._registry:
            raise ValueError(f"No runnable registered for class type: {cls_type}")
        return self._registry[cls_type]()

    def register(self, class_type: K, runnable_cls: Callable[[], R]) -> None:
        self._registry[class_type] = runnable_cls
