# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import statistics
from datetime import UTC, datetime, timedelta
from multiprocessing.synchronize import Condition
from uuid import UUID

from sqlalchemy.orm import Session

from app.db import get_db_session
from app.repositories import PipelineRepository
from app.schemas import Pipeline, PipelineStatus
from app.schemas.metrics import InferenceMetrics, LatencyMetrics, PipelineMetrics, TimeWindow
from app.services import ActivePipelineService
from app.services.base import (
    GenericPersistenceService,
    ResourceInUseError,
    ResourceNotFoundError,
    ResourceType,
    ServiceConfig,
)
from app.services.mappers import PipelineMapper
from app.services.metrics_service import MetricsService
from app.services.parent_process_guard import parent_process_only

MSG_ERR_DELETE_RUNNING_PIPELINE = "Cannot delete a running pipeline."


class PipelineService:
    def __init__(
        self,
        active_pipeline_service: ActivePipelineService,
        metrics_service: MetricsService,
        config_changed_condition: Condition,
    ) -> None:
        self._persistence: GenericPersistenceService[Pipeline, PipelineRepository] = GenericPersistenceService(
            ServiceConfig(PipelineRepository, PipelineMapper, ResourceType.PIPELINE)
        )
        self._active_pipeline_service: ActivePipelineService = active_pipeline_service
        self._config_changed_condition: Condition = config_changed_condition
        self._metrics_service: MetricsService = metrics_service

    def _notify_source_changed(self) -> None:
        with self._config_changed_condition:
            self._config_changed_condition.notify_all()

    def _notify_sink_changed(self) -> None:
        self._active_pipeline_service.reload()

    def _notify_pipeline_changed(self) -> None:
        self._notify_source_changed()
        self._notify_sink_changed()

    def get_pipeline_by_id(self, pipeline_id: UUID, db: Session | None = None) -> Pipeline:
        """Retrieve a pipeline by its ID."""
        pipeline = self._persistence.get_by_id(pipeline_id, db)
        if not pipeline:
            raise ResourceNotFoundError(ResourceType.PIPELINE, str(pipeline_id))
        return pipeline

    def list_pipelines(self) -> list[Pipeline]:
        """List all pipelines."""
        return self._persistence.list_all()

    @parent_process_only
    def create_pipeline(self, pipeline: Pipeline) -> Pipeline:
        """Create a new pipeline."""
        with get_db_session() as db:
            created = self._persistence.create(pipeline, db)
            db.commit()
            if created.status == PipelineStatus.RUNNING:
                self._notify_pipeline_changed()
            return created

    @parent_process_only
    def update_pipeline(self, pipeline_id: UUID, partial_config: dict) -> Pipeline:
        """Update an existing pipeline."""
        with get_db_session() as db:
            pipeline = self.get_pipeline_by_id(pipeline_id, db)
            updated = self._persistence.update(pipeline, partial_config, db)
            db.commit()
            if pipeline.status == PipelineStatus.RUNNING and updated.status == PipelineStatus.RUNNING:
                # If the pipeline source_id or sink_id is being updated while running
                if pipeline.source_id != updated.source_id:
                    self._notify_source_changed()
                if pipeline.sink_id != updated.sink_id:
                    self._notify_sink_changed()
            elif pipeline.status != updated.status:
                # If the pipeline is being activated or stopped
                self._notify_pipeline_changed()
            return updated

    @parent_process_only
    def delete_pipeline_by_id(self, pipeline_id: UUID) -> None:
        """Delete a pipeline by its ID."""
        with get_db_session() as db:
            pipeline = self.get_pipeline_by_id(pipeline_id, db)
            if pipeline.status == PipelineStatus.RUNNING:
                raise ResourceInUseError(ResourceType.PIPELINE, str(pipeline_id), MSG_ERR_DELETE_RUNNING_PIPELINE)
            self._persistence.delete_by_id(pipeline_id, db)
            db.commit()

    def get_pipeline_metrics(self, pipeline_id: UUID, time_window: int = 60) -> PipelineMetrics:
        """Calculate metrics for a pipeline over a specified time window."""
        # First check if pipeline exists
        pipeline = self.get_pipeline_by_id(pipeline_id)
        if pipeline.status != PipelineStatus.RUNNING:
            raise ValueError("Cannot get metrics for a pipeline that is not running.")

        # Calculate time window
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(seconds=time_window)

        # Get actual latency measurements from the metrics service
        latency_samples = self._metrics_service.get_latency_measurements(
            model_id=pipeline.model_id,  # type: ignore[arg-type]
            time_window=time_window,
        )

        # Calculate latency metrics
        if latency_samples:
            latency_metrics = LatencyMetrics(
                avg_ms=statistics.mean(latency_samples),
                min_ms=min(latency_samples),
                max_ms=max(latency_samples),
                p95_ms=self._calculate_percentile(latency_samples, 95),
                latest_ms=latency_samples[-1],
            )
        else:
            # No data available
            latency_metrics = LatencyMetrics(avg_ms=None, min_ms=None, max_ms=None, p95_ms=None, latest_ms=None)

        window = TimeWindow(start=start_time, end=end_time, time_window=time_window)
        inference_metrics = InferenceMetrics(latency=latency_metrics)
        return PipelineMetrics(time_window=window, inference=inference_metrics)

    @staticmethod
    def _calculate_percentile(data: list[float], percentile: int) -> float:
        """Calculate the specified percentile of the data."""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (percentile / 100.0)
        floor_k = int(k)
        ceil_k = floor_k + 1

        if ceil_k >= len(sorted_data):
            return sorted_data[-1]

        # Linear interpolation
        fraction = k - floor_k
        return sorted_data[floor_k] + fraction * (sorted_data[ceil_k] - sorted_data[floor_k])
