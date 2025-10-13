# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

ReportFn = Callable[[float], None]
HeartbeatFn = Callable[[], None]
T = TypeVar("T")


@dataclass(frozen=True, kw_only=True, slots=True)
class ExecutionContext[T]:
    task: T
    report_progress: ReportFn
    heartbeat: HeartbeatFn


TContext = TypeVar("TContext", bound=ExecutionContext)
TContext_contra = TypeVar("TContext_contra", contravariant=True, bound=ExecutionContext)


class Runnable(Protocol[TContext_contra]):  # ignore
    """Generic interface for activities executed by runners."""

    def run(self, ctx: TContext_contra) -> None: ...


class RunnableFactory(Generic[TContext]):
    """Factory protocol for creating Runnable instances."""

    def __init__(self, runnable_cls: type[Runnable[TContext]]) -> None:
        self._runnable_cls = runnable_cls

    def __call__(self) -> Runnable[TContext]:
        return self._runnable_cls()
