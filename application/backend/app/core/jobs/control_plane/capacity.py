# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import TracebackType


class _Permit:
    def __init__(self, sem: asyncio.Semaphore):
        self._sem = sem

    async def __aenter__(self):
        await self._sem.acquire()

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ):
        self._sem.release()


class Capacity:
    def __init__(self, n: int):
        self._sem = asyncio.Semaphore(max(1, n))

    def permit(self) -> _Permit:
        return _Permit(self._sem)
