# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .base import Trainer, TrainerContext


class OTXTrainer(Trainer):
    def run(self, ctx: TrainerContext) -> None:
        raise NotImplementedError
