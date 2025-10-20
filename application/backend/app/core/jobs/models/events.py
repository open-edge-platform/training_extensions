# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    pass


@dataclass(frozen=True, slots=True)
class Started(ExecutionEvent):
    pass


@dataclass(frozen=True, slots=True)
class Progress(ExecutionEvent):
    message: str
    value: float  # percentage of completion [0.0, 100.0]


@dataclass(frozen=True, slots=True)
class Done(ExecutionEvent):
    pass


@dataclass(frozen=True, slots=True)
class Cancelled(ExecutionEvent):
    pass


@dataclass(frozen=True, slots=True)
class Failed(ExecutionEvent):
    details: str
