# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC

from app.core.run import ExecutionContext, Runnable

from .models import TrainingParams


class Trainer(Runnable, ABC):
    @staticmethod
    def get_training_params(ctx: ExecutionContext) -> TrainingParams:
        return TrainingParams.model_validate_json(ctx.payload)
