# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from app.core.jobs.control_plane.capacity import Capacity, _Permit


class TestCapacity:
    """Test cases for Capacity management."""

    @pytest.mark.parametrize("initial_value,expected_value", [(5, 5), (0, 1), (-3, 1)])
    def test_capacity_init(self, initial_value, expected_value):
        """Test Capacity initialization with different values."""
        capacity = Capacity(initial_value)
        assert capacity._sem._value == expected_value

    def test_permit_returns_permit_instance(self):
        """Test permit method returns _Permit instance."""
        capacity = Capacity(1)
        permit = capacity.permit()
        assert isinstance(permit, _Permit)

    @pytest.mark.asyncio
    async def test_capacity_limits_concurrent_access(self):
        """Test capacity properly limits concurrent access."""
        capacity = Capacity(2)

        acquired_count = 0
        max_concurrent = 0

        async def use_capacity():
            nonlocal acquired_count, max_concurrent
            async with capacity.permit():
                acquired_count += 1
                max_concurrent = max(max_concurrent, acquired_count)
                await asyncio.sleep(0.1)
                acquired_count -= 1

        # Start 5 tasks with capacity of 2
        tasks = [asyncio.create_task(use_capacity()) for _ in range(5)]
        await asyncio.gather(*tasks)

        # Maximum concurrent should not exceed capacity
        assert max_concurrent <= 2
        assert acquired_count == 0  # All should be released

    @pytest.mark.asyncio
    async def test_capacity_single_permit(self):
        """Test capacity with single permit works correctly."""
        capacity = Capacity(1)

        execution_order = []

        async def sequential_task(task_id):
            async with capacity.permit():
                execution_order.append(f"start_{task_id}")
                await asyncio.sleep(0.05)
                execution_order.append(f"end_{task_id}")

        # Start multiple tasks
        tasks = [asyncio.create_task(sequential_task(i)) for i in range(3)]
        await asyncio.gather(*tasks)

        # Tasks should execute sequentially
        assert len(execution_order) == 6
        # Each task should complete before the next starts
        for i in range(3):
            start_idx = execution_order.index(f"start_{i}")
            end_idx = execution_order.index(f"end_{i}")
            assert start_idx < end_idx

    @pytest.mark.asyncio
    async def test_capacity_full_utilization(self):
        """Test capacity allows full utilization up to limit."""
        capacity = Capacity(3)

        running_tasks = []

        async def long_running_task():
            async with capacity.permit():
                running_tasks.append(asyncio.current_task())
                await asyncio.sleep(0.2)
                running_tasks.remove(asyncio.current_task())

        # Start tasks and check concurrent execution
        tasks = [asyncio.create_task(long_running_task()) for _ in range(3)]

        # Give tasks time to start
        await asyncio.sleep(0.1)

        # All 3 should be running concurrently
        assert len(running_tasks) == 3

        await asyncio.gather(*tasks)

        # All should be done
        assert len(running_tasks) == 0
