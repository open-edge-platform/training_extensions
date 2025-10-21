# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import statistics
from datetime import UTC, datetime, timedelta
from multiprocessing.synchronize import Condition
from uuid import UUID

from sqlalchemy.orm import Session

from app.repositories import PipelineRepository
from app.schemas import PipelineStatus, PipelineView
from app.schemas.metrics import InferenceMetrics, LatencyMetrics, PipelineMetrics, ThroughputMetrics, TimeWindow

from .active_pipeline_service import ActivePipelineService
from .base import ResourceNotFoundError, ResourceType
from .data_collect import DataCollector
from .mappers import PipelineMapper
from .metrics_service import MetricsService
from .parent_process_guard import parent_process_only

MSG_ERR_DELETE_RUNNING_PIPELINE = "Cannot delete a running pipeline."


class PipelineService:
    def __init__(
        self,
        active_pipeline_service: ActivePipelineService,
        data_collector: DataCollector,
        metrics_service: MetricsService,
        config_changed_condition: Condition,
        db_session: Session,
    ) -> None:
        self._active_pipeline_service: ActivePipelineService = active_pipeline_service
        self._data_collector: DataCollector = data_collector
        self._config_changed_condition: Condition = config_changed_condition
        self._metrics_service: MetricsService = metrics_service
        self._db_session: Session = db_session

    def _notify_source_changed(self) -> None:
        with self._config_changed_condition:
            self._config_changed_condition.notify_all()

    def _notify_sink_changed(self) -> None:
        self._active_pipeline_service.reload()

    def _notify_pipeline_changed(self) -> None:
        self._notify_source_changed()
        self._notify_sink_changed()
        self._data_collector.reload_policies()

    def get_pipeline_by_id(self, project_id: UUID) -> PipelineView:
        """Retrieve a pipeline by project ID."""
        pipeline_repo = PipelineRepository(self._db_session)
        pipeline = pipeline_repo.get_by_id(str(project_id))
        if not pipeline:
            raise ResourceNotFoundError(ResourceType.PIPELINE, str(project_id))
        return PipelineMapper.to_schema(pipeline)

    @parent_process_only
    def update_pipeline(self, project_id: UUID, partial_config: dict) -> PipelineView:
        """Update an existing pipeline."""
        pipeline = self.get_pipeline_by_id(project_id)
        to_update = type(pipeline).model_validate(pipeline.model_copy(update=partial_config))
        pipeline_repo = PipelineRepository(self._db_session)
        updated = PipelineMapper.to_schema(pipeline_repo.update(PipelineMapper.from_schema(to_update)))
        if pipeline.status == PipelineStatus.RUNNING and updated.status == PipelineStatus.RUNNING:
            # If the pipeline source_id or sink_id is being updated while running
            if pipeline.source.id != updated.source.id:  # type: ignore[union-attr] # source is always there for running pipeline
                self._notify_source_changed()
            if pipeline.sink.id != updated.sink.id:  # type: ignore[union-attr] # sink is always there for running pipeline
                self._notify_sink_changed()
            if pipeline.data_collection_policies != updated.data_collection_policies:
                self._active_pipeline_service.reload()
                self._data_collector.reload_policies()
        elif pipeline.status != updated.status:
            # If the pipeline is being activated or stopped
            self._notify_pipeline_changed()
        return updated

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
            model_id=pipeline.model_id,  # type: ignore[arg-type] # model is always there for running pipeline
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

        # Get throughput measurements from the metrics service
        total_requests, throughput_data = self._metrics_service.get_throughput_measurements(
            model_id=pipeline.model_id,  # type: ignore[arg-type]
            time_window=time_window,
        )
        if total_requests:
            throughput_metrics = ThroughputMetrics(
                avg_requests_per_second=total_requests / time_window if time_window > 0 else 0.0,
                total_requests=total_requests,
                max_requests_per_second=max((count for _, count in throughput_data), default=0),
            )
        else:
            # No data available
            throughput_metrics = ThroughputMetrics(
                avg_requests_per_second=None, total_requests=None, max_requests_per_second=None
            )

        window = TimeWindow(start=start_time, end=end_time, time_window=time_window)
        inference_metrics = InferenceMetrics(latency=latency_metrics, throughput=throughput_metrics)
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
