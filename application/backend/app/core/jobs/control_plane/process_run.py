# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import logging
import multiprocessing as mp
from collections.abc import Iterator
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnProcess
from multiprocessing.synchronize import Event
from pathlib import Path

from app.core.jobs.models import Done, ExecutionEvent, Failed, Job, JobType, Started
from app.core.run import ExecutionContext, RunnableFactory, Runner

logger = logging.getLogger(__name__)


class ProcessRun:
    """Owns the child Process+IPC and translates to domain events."""

    def __init__(self, ctx: mp.context.SpawnContext, data_dir: Path, runnable_factory: RunnableFactory, job: Job):
        self._ctx = ctx
        self._data_dir = data_dir
        self._runnable_factory = runnable_factory
        self._job = job
        self._parent, self._child = ctx.Pipe(duplex=False)
        self._cancel = ctx.Event()
        self._proc: SpawnProcess | None = None

    def start(self) -> "ProcessRun":
        self._proc = self._ctx.Process(
            target=_entrypoint,
            args=(
                self._runnable_factory,
                self._data_dir,
                self._job.job_type,
                self._job.params.model_dump_json(),
                self._child,
                self._cancel,
            ),
            name=f"trainer-{self._job.id}",
        )
        self._proc.start()
        self._child.close()
        return self

    def events(self) -> Iterator[ExecutionEvent]:
        """Blocking iterator; the control plane decides how to multiplex."""
        try:
            while True:
                msg = self._parent.recv()  # blocks
                yield msg
        except EOFError:
            # Child exited; infer outcome
            code = self._proc.exitcode if self._proc else 1
            yield Done() if code == 0 else Failed(f"process exit {code}")
        finally:
            self._parent.close()

    async def stop(self, graceful_timeout: float = 6.0, term_timeout: float = 3.0, kill_timeout: float = 1.0) -> None:
        """
        Stop the process with graceful degradation.

        Args:
            graceful_timeout: How long to wait for graceful shutdown (seconds)
            term_timeout: How long to wait after SIGTERM (seconds)
            kill_timeout: How long to wait after SIGKILL (seconds)
        """
        if self._proc is None:
            return

        # Request graceful shutdown
        self._cancel.set()
        await asyncio.to_thread(self._proc.join, timeout=graceful_timeout)
        if not self._proc.is_alive():
            logger.debug("Process %s stopped gracefully", self._proc.name)
            return

        # Try SIGTERM
        self._proc.terminate()
        await asyncio.to_thread(self._proc.join, timeout=term_timeout)
        if not self._proc.is_alive():
            logger.debug("Process %s terminated gracefully", self._proc.name)
            return

        # Last resort: SIGKILL
        logger.warning("Force killing process %s", self._proc.name)
        self._proc.kill()
        await asyncio.to_thread(self._proc.join, timeout=kill_timeout)
        if self._proc.is_alive():
            logger.error("Process %s doesn't respond to SIGKILL", self._proc.name)


def _entrypoint(
    get_runnable: RunnableFactory, data_dir: Path, job_type: str, payload: str, conn: Connection, cancel_event: Event
) -> None:
    import traceback

    from app.core.jobs.models import Cancelled, Done, Failed, Progress

    class CancelledExc(Exception):
        pass

    def report(msg: str, p: float) -> None:
        conn.send(Progress(message=msg, value=p))

    # alt: another possible solution is to run the heartbeat in a separate daemon thread at a set interval, so it is not
    # coupled to the training process.
    def heartbeat():
        if cancel_event.is_set():
            raise CancelledExc

    runnable = get_runnable(JobType(job_type))

    try:
        conn.send(Started())
        runnable.run(ExecutionContext(payload=payload, data_dir=data_dir, report=report, heartbeat=heartbeat))
        conn.send(Done())
    except CancelledExc:
        conn.send(Cancelled())
    except Exception:
        conn.send(Failed(traceback.format_exc()))
    finally:
        with contextlib.suppress(Exception):
            conn.close()


class ProcessRunnerFactory:
    """Process-based infra with spawned context"""

    def __init__(self, data_dir: Path, runnable_factory: RunnableFactory) -> None:
        # consider using native context for python 3.14 due to upgrade to 'fork_server' model
        self._ctx = mp.get_context("spawn")
        self._data_dir = data_dir
        self._runnable_factory = runnable_factory

    def for_job(self, job: Job) -> Runner[Job, ExecutionEvent]:
        return ProcessRun(self._ctx, self._data_dir, self._runnable_factory, job)
