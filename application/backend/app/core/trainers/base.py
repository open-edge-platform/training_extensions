# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.jobs.models import Job
from app.core.run import ExecutionContext


@dataclass(frozen=True, kw_only=True, slots=True)
class TrainerContext(ExecutionContext[Job]):
    pass


class Trainer(ABC):
    @abstractmethod
    def run(self, ctx: TrainerContext) -> None: ...
