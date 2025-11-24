# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
from unittest.mock import Mock, patch

import pytest

from app.core.jobs.exec import ProcessRun
from app.core.jobs.models import Job, Progress
from app.core.jobs.models.events import Done, Failed, Started
from app.core.run import RunnableFactory


class TestProcessRun:
    """Test cases for ProcessRun functionality."""

    @pytest.fixture
    def fxt_context(self):
        ctx = Mock(spec=mp.context.SpawnContext)
        parent_conn = Mock()
        child_conn = Mock()
        ctx.Pipe.return_value = (parent_conn, child_conn)
        ctx.Event.return_value = Mock()
        ctx.Process.return_value = Mock()
        return ctx

    @pytest.fixture
    def fxt_runnable_factory(self):
        return Mock(spec=RunnableFactory)

    @pytest.fixture
    def fxt_process_run(self, fxt_context, fxt_runnable_factory, fxt_job):
        return ProcessRun(fxt_context, fxt_runnable_factory, fxt_job())

    def test_init(self, fxt_context, fxt_runnable_factory, fxt_job):
        process_run = ProcessRun(fxt_context, fxt_runnable_factory, fxt_job())

        assert process_run._ctx == fxt_context
        assert process_run._runnable_factory == fxt_runnable_factory
        assert isinstance(process_run._job, Job)
        assert process_run._proc is None

        # Verify pipe and event creation
        fxt_context.Pipe.assert_called_once_with(duplex=False)
        fxt_context.Event.assert_called_once()

    def test_start_creates_and_starts_process(self, fxt_process_run, fxt_context):
        """Test start method creates and starts process."""
        mock_process = Mock()
        fxt_context.Process.return_value = mock_process

        result = fxt_process_run.start()

        assert result == fxt_process_run
        assert fxt_process_run._proc == mock_process
        mock_process.start.assert_called_once()
        fxt_process_run._child.close.assert_called_once()

    def test_events_yields_messages_from_parent_connection(self, fxt_process_run):
        """Test events method yields messages from parent connection."""
        # Mock messages from parent connection
        messages = [Started(), Progress("training in progress", 50.0)]
        fxt_process_run._parent.recv.side_effect = [*messages, EOFError()]

        # Mock process with successful exit
        mock_process = Mock()
        mock_process.exitcode = 0
        fxt_process_run._proc = mock_process

        events = list(fxt_process_run.events())

        # Should yield the messages plus a final Done event for successful exit
        assert len(events) == len(messages) + 1
        assert events[0] == messages[0]
        assert events[1] == messages[1]
        assert events[2] == Done()

    def test_events_handles_eof_error_with_failure_exit(self, fxt_process_run):
        """Test events method handles EOFError with failed process exit."""
        fxt_process_run._parent.recv.side_effect = EOFError()

        mock_process = Mock()
        mock_process.exitcode = 1
        fxt_process_run._proc = mock_process

        events = list(fxt_process_run.events())

        # Should yield Failed event for failed exit
        assert len(events) == 1
        assert events[0] == Failed("process exit 1")

    def test_events_handles_no_process(self, fxt_process_run):
        """Test events method handles case where process is None."""
        fxt_process_run._parent.recv.side_effect = EOFError()
        fxt_process_run._proc = None

        events = list(fxt_process_run.events())

        # Should yield Failed event when no process
        assert len(events) == 1
        assert events[0] == Failed("process exit 1")

    def test_events_closes_parent_connection(self, fxt_process_run):
        """Test events method closes parent connection when done."""
        fxt_process_run._parent.recv.side_effect = EOFError()
        fxt_process_run._proc = Mock(exitcode=0)

        list(fxt_process_run.events())  # Consume all events

        fxt_process_run._parent.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_with_no_process(self, fxt_process_run):
        """Test stop method when no process exists."""
        fxt_process_run._proc = None

        # Should complete without error
        await fxt_process_run.stop()

    @pytest.mark.asyncio
    async def test_stop_graceful_shutdown(self, fxt_process_run):
        """Test stop method with graceful shutdown."""
        mock_process = Mock()
        mock_process.is_alive.return_value = False
        fxt_process_run._proc = mock_process

        await fxt_process_run.stop()

        # Should set cancel event and try graceful join
        fxt_process_run._cancel.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_with_terminate(self, fxt_process_run):
        """Test stop method escalates to terminate."""
        mock_process = Mock()
        # First call (after graceful timeout) returns True, second (after terminate) returns False
        mock_process.is_alive.side_effect = [True, False]
        fxt_process_run._proc = mock_process

        with patch("asyncio.to_thread") as mock_to_thread:
            # Mock join calls
            mock_to_thread.return_value = None

            await fxt_process_run.stop(graceful_timeout=0.01, term_timeout=0.01)

            # Should call terminate after graceful timeout
            mock_process.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_with_kill(self, fxt_process_run):
        """Test stop method escalates to kill."""
        mock_process = Mock()
        # Always alive until kill
        mock_process.is_alive.side_effect = [True, True, False]
        fxt_process_run._proc = mock_process

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = None

            with patch("app.core.jobs.exec.process_run.logger") as mock_logger:
                await fxt_process_run.stop(graceful_timeout=0.01, term_timeout=0.01, kill_timeout=0.01)

                # Should call both terminate and kill
                mock_process.terminate.assert_called_once()
                mock_process.kill.assert_called_once()
                mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_stop_process_survives_kill(self, fxt_process_run):
        """Test stop method logs error if process survives kill."""
        mock_process = Mock()
        mock_process.name = "test-process"
        # Always alive
        mock_process.is_alive.return_value = True
        fxt_process_run._proc = mock_process

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = None

            with patch("app.core.jobs.exec.process_run.logger") as mock_logger:
                await fxt_process_run.stop(graceful_timeout=0.01, term_timeout=0.01, kill_timeout=0.01)

                # Should log error about unkillable process
                mock_logger.error.assert_called_with("Process {} doesn't respond to SIGKILL", "test-process")
