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
from app.schemas import Project
from app.schemas.dataset_item import DatasetItemFormat
from app.schemas.pipeline import FixedRateDataCollectionPolicy
from app.services import ActivePipelineService, DatasetService
from app.services.data_collect.prediction_converter import convert_prediction

logger = logging.getLogger(__name__)


class PolicyChecker(metaclass=ABCMeta):
    @abstractmethod
    def should_collect(self, timestamp: float) -> bool:
        """
        Checks if certain dispatched image should be collected to the project dataset

        :param timestamp: dispatched image timestamp
        :return: True if image should be collected, False otherwise
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
        project: Project,
        timestamp: float,
        frame_data: np.ndarray,
        inference_data: InferenceData,
    ) -> None:
        """
        Collects the dispatched image if any of the policy checkers indicate that it should be collected.

        :param source_id: ID of the pipeline source
        :param project: Pipeline project
        :param timestamp: Timestamp of the image
        :param frame_data: Image binary data in ndarray format
        :param inference_data: Inference data including predictions
        :return:
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
