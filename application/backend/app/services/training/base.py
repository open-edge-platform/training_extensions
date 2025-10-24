# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from app.core.run import ExecutionContext, Runnable

from .models import TrainingParams


class PipelineContext:
    """Context for the training pipeline execution."""

    def __init__(self) -> None:
        self.data: dict[str, object] = {}

    def set(self, key: str, value: object) -> None:
        self.data[key] = value

    def get(self, key: str, default: object = None) -> object:
        return self.data.get(key, default)


class TrainingStep(ABC):
    """A single step in the training pipeline."""

    @abstractmethod
    def execute(self, ctx: ExecutionContext, params: TrainingParams, pipeline_ctx: PipelineContext) -> None:
        """Execute this training step."""

    @abstractmethod
    def get_name(self) -> str:
        """Return human-readable step name."""


class TrainingPipeline:
    """Composable pipeline of training steps."""

    def __init__(self, steps: list[TrainingStep]):
        self._steps = steps

    def execute(self, ctx: ExecutionContext, params: TrainingParams) -> None:
        pipeline_ctx = PipelineContext()
        for step in self._steps:
            ctx.report_progress(f"Starting: {step.get_name()}")
            step.execute(ctx, params, pipeline_ctx)
            ctx.report_progress(f"Completed: {step.get_name()}")
            ctx.heartbeat()


class Trainer(Runnable, ABC):
    """
    Abstract base class for model training workflows.

    Subclasses should build a training pipeline by composing
    appropriate training steps in the _build_pipeline method.
    """

    def __init__(self) -> None:
        self._pipeline: TrainingPipeline | None = None

    @abstractmethod
    def _build_pipeline(self) -> TrainingPipeline:
        """
        Build the training pipeline with all necessary steps.

        Returns:
            Configured training pipeline
        """

    def run(self, ctx: ExecutionContext) -> None:
        """
        Execute the training workflow.

        Args:
            ctx: Execution context with job parameters
        """
        params = self._get_training_params(ctx)
        ctx.report_progress(f"Training job started: {params.job_id}")

        if self._pipeline is None:
            self._pipeline = self._build_pipeline()

        self._pipeline.execute(ctx, params)

        ctx.report_progress(f"Training job completed: {params.job_id}")

    @staticmethod
    def _get_training_params(ctx: ExecutionContext) -> TrainingParams:
        return TrainingParams.model_validate_json(ctx.payload)
