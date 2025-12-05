# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from abc import ABCMeta, abstractmethod
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from app.db import get_db_session
from app.models import (
    ConfidenceThresholdDataCollectionPolicy,
    DatasetItemFormat,
    FixedRateDataCollectionPolicy,
    Pipeline,
    Project,
)
from app.services.data_collect.prediction_converter import convert_prediction, get_confidence_scores
from app.services.event.event_bus import EventBus, EventType
from app.stream.stream_data import InferenceData


class PolicyChecker(metaclass=ABCMeta):
    @abstractmethod
    def should_collect(self, timestamp: float, confidence_scores: list[float]) -> bool:
        """
        Determines whether a dispatched image should be collected to the project dataset.

        Args:
            timestamp: Floating-point timestamp representing when the image was
                    dispatched, in seconds since epoch.
            confidence_scores: Floating-point timestamp predictions confidence scores.

        Returns:
            bool: True if the image meets the collection criteria and should be
                added to the dataset, False otherwise.

        Note:
            This is an abstract method and must be implemented by subclasses.
            Different policy implementations may use different strategies such as
            time-based sampling, random sampling, or condition-based triggering.
        """


class FixedRatePolicyChecker(PolicyChecker):
    """
    Fixed rate data collection policy checker.
    Checks when the last image has been collected against collection rate.
    """

    def __init__(self, policy: FixedRateDataCollectionPolicy):
        self.min_interval = 1.0 / policy.rate
        self.last_collect_time = 0.0

    def should_collect(self, timestamp: float, confidence_scores: list[float]) -> bool:  # noqa: ARG002
        time_since_last = timestamp - self.last_collect_time
        if time_since_last < self.min_interval:
            return False
        self.last_collect_time = timestamp
        return True


class ConfidenceThresholdPolicyChecker(PolicyChecker):
    """
    Confidence threshold data collection policy checker.
    Collects only images where any prediction confidence is lower than predefined threshold.
    To prevent the collection of multiple, almost identical frames in rapid succession, it also checks image timestamp
    with min_sampling_interval.
    """

    def __init__(self, policy: ConfidenceThresholdDataCollectionPolicy):
        self.confidence_threshold = policy.confidence_threshold
        self.min_sampling_interval = policy.min_sampling_interval
        self.last_collect_time = 0.0

    def should_collect(self, timestamp: float, confidence_scores: list[float]) -> bool:
        time_since_last = timestamp - self.last_collect_time

        if (
            all(confidence >= self.confidence_threshold for confidence in confidence_scores)
            or time_since_last < self.min_sampling_interval
        ):
            return False
        self.last_collect_time = timestamp
        return True


class DataCollector:
    def __init__(self, data_dir: Path, event_bus: EventBus) -> None:
        self.should_collect_next_frame = False
        self.data_dir = data_dir
        self.event_bus = event_bus
        self.active_pipeline_data: tuple[Pipeline, Project] | None = None
        self.policy_checkers: list[PolicyChecker] = []

        self._load_pipeline()
        event_bus.subscribe(
            [
                EventType.PIPELINE_STATUS_CHANGED,
                EventType.PIPELINE_DATASET_COLLECTION_POLICIES_CHANGED,
                EventType.SOURCE_CHANGED,
            ],
            self._load_pipeline,
        )

    def _load_pipeline(self) -> None:
        from app.services import LabelService, PipelineService, ProjectService

        with get_db_session() as db:
            label_service = LabelService(db_session=db)
            pipeline_service = PipelineService(event_bus=self.event_bus, db_session=db)
            pipeline = pipeline_service.get_active_pipeline()
            if pipeline is None:
                logger.info("No active pipeline found, disabling data collection")
                self.active_pipeline_data = None
                self.policy_checkers = []
                return

            project_service = ProjectService(
                data_dir=self.data_dir, db_session=db, label_service=label_service, pipeline_service=pipeline_service
            )
            project = project_service.get_project_by_id(pipeline.project_id)

            self.active_pipeline_data = pipeline, project
            logger.info(
                "Dataset collection policies set to {}, source: {}",
                pipeline.data_collection_policies,
                pipeline.source_id,
            )

            policies = [policy for policy in pipeline.data_collection_policies if policy.enabled]
            self.policy_checkers = []
            for policy in policies:
                checker: PolicyChecker | None = None
                match policy:
                    case FixedRateDataCollectionPolicy():
                        checker = FixedRatePolicyChecker(policy)
                    case ConfidenceThresholdDataCollectionPolicy():
                        checker = ConfidenceThresholdPolicyChecker(policy)
                if checker is not None:
                    self.policy_checkers.append(checker)

    def collect(
        self,
        timestamp: float,
        frame_data: np.ndarray,
        inference_data: InferenceData,
    ) -> None:
        """
        Collects dispatched images to project dataset based on policy checkers.

        Evaluates automated collection policies and the manual next-frame trigger to determine if image
        should be added to dataset. If collection is warranted, processes the image and
        creates a dataset item with annotations.

        Args:
            timestamp: Floating-point timestamp of the captured image, used for item naming.
            confidence: Floating-point confidence of the captured image, used for item naming.
            frame_data: Image data in numpy ndarray format (expected in BGR color space).
            inference_data: Inference data containing model predictions and model identifier.

        Returns:
            None: Method performs operations with side effects but returns no value.

        Note:
            Collection occurs if any policy checker returns True OR if the
            should_collect_next_frame flag is set. Timestamp is formatted to string
            with 4 decimal places for use as dataset item name.
        """
        if self.active_pipeline_data is None:
            return
        pipeline, project = self.active_pipeline_data
        from app.services import DatasetService, LabelService

        confidence_scores = get_confidence_scores(prediction=inference_data.prediction)
        should_collect = (
            any(
                checker.should_collect(timestamp=timestamp, confidence_scores=confidence_scores)
                for checker in self.policy_checkers
            )
            or self.should_collect_next_frame
        )
        if not should_collect:
            return
        frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        with get_db_session() as session:
            label_service = LabelService(db_session=session)
            labels = label_service.list_all(project_id=project.id)
            annotations = convert_prediction(labels=labels, frame_data=frame_data, prediction=inference_data.prediction)

            dataset_service = DatasetService(data_dir=self.data_dir, label_service=label_service, db_session=session)
            dataset_service.create_dataset_item(
                project=project,
                name=f"{timestamp:.4f}".replace(".", "_"),
                format=DatasetItemFormat.JPG,
                data=frame_data,
                user_reviewed=False,
                source_id=pipeline.source_id,
                prediction_model_id=inference_data.model_id,
                annotations=annotations,
            )
        self.should_collect_next_frame = False

    def collect_next_frame(self) -> None:
        """
        Sets flag to collect the next available frame. This flag will be reset to False upon successful collection.
        """
        self.should_collect_next_frame = True
