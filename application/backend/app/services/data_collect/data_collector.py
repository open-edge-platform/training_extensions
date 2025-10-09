# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path
from uuid import UUID

import cv2
import numpy as np

from app.db import get_db_session
from app.entities.stream_data import InferenceData
from app.schemas import ProjectView
from app.schemas.dataset_item import DatasetItemFormat
from app.schemas.pipeline import FixedRateDataCollectionPolicy
from app.services import ActivePipelineService, DatasetService
from app.services.data_collect.prediction_converter import convert_prediction

logger = logging.getLogger(__name__)


class PolicyChecker(metaclass=ABCMeta):
    @abstractmethod
    def should_collect(self, timestamp: float) -> bool:
        """
        Determines whether a dispatched image should be collected to the project dataset.

        Args:
            timestamp: Floating-point timestamp representing when the image was
                    dispatched, in seconds since epoch.

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

    def should_collect(self, timestamp: float) -> bool:
        time_since_last = timestamp - self.last_collect_time
        if time_since_last < self.min_interval:
            return False
        self.last_collect_time = timestamp
        return True


class DataCollector:
    def __init__(self, data_dir: Path, active_pipeline_service: ActivePipelineService) -> None:
        super().__init__()
        self.should_collect_next_frame = False
        self.data_dir = data_dir
        self.active_pipeline_service = active_pipeline_service
        self.policy_checkers: list[PolicyChecker] = []
        self.reload_policies()

    def collect(
        self,
        source_id: UUID,
        project: ProjectView,
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
            source_id: UUID identifying the pipeline source that generated the image.
            project: Project object representing the target project for dataset collection.
            timestamp: Floating-point timestamp of the captured image, used for item naming.
            frame_data: Image data in numpy ndarray format (expected in BGR color space).
            inference_data: Inference data containing model predictions and model identifier.

        Returns:
            None: Method performs operations with side effects but returns no value.

        Note:
            Collection occurs if any policy checker returns True OR if the
            should_collect_next_frame flag is set. Timestamp is formatted to string
            with 4 decimal places for use as dataset item name.
        """
        should_collect = (
            any(checker.should_collect(timestamp) for checker in self.policy_checkers) or self.should_collect_next_frame
        )
        if not should_collect:
            return
        frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        annotations = convert_prediction(
            labels=project.task.labels, frame_data=frame_data, prediction=inference_data.prediction
        )
        with get_db_session() as session:
            dataset_service = DatasetService(self.data_dir, session)
            dataset_service.create_dataset_item(
                project_id=project.id,
                name=f"{timestamp:.4f}".replace(".", "_"),
                format=DatasetItemFormat.JPG,
                data=frame_data,
                user_reviewed=False,
                source_id=source_id,
                prediction_model_id=inference_data.model_id,
                annotations=annotations,
            )
        self.should_collect_next_frame = False

    def collect_next_frame(self) -> None:
        """
        Sets flag to collect the next available frame. This flag will be reset to False upon successful collection.
        """
        self.should_collect_next_frame = True

    def reload_policies(self) -> None:
        """
        Reloads data collection policies from active pipeline and re-initialize policy checkers.
        """
        policies = [policy for policy in self.active_pipeline_service.get_data_collection_policies() if policy.enabled]
        self.policy_checkers = []
        for policy in policies:
            manager = None
            match policy:
                case FixedRateDataCollectionPolicy():
                    manager = FixedRatePolicyChecker(policy)
            if manager is not None:
                self.policy_checkers.append(manager)
