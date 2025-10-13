# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from typing import Protocol, TypeVar

T = TypeVar("T")  # Task type
E = TypeVar("E")  # Event type


class Runner(Protocol[T, E]):
    """
    Protocol for a task runner.

    Represents an interface for starting, stopping, and monitoring the progress of an isolated workload.
    """

    def start(self) -> "Runner[T, E]": ...
    def events(self) -> Iterator[E]: ...
    async def stop(
        self, graceful_timeout: float = 6.0, term_timeout: float = 3.0, kill_timeout: float = 1.0
    ) -> None: ...


class RunnerFactory(Protocol[T, E]):
    """
    Protocol for a factory that creates Runner instances for specific task types.
    """

    def for_job(self, task: T) -> Runner[T, E]: ...
