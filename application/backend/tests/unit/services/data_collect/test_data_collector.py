# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
import time_machine

from app.models import DatasetItemAnnotation, DatasetItemFormat, FullImage, Label, LabelReference
from app.schemas.pipeline import ConfidenceThresholdDataCollectionPolicy, FixedRateDataCollectionPolicy
from app.services import DatasetService, LabelService
from app.services.data_collect.data_collector import (
    ConfidenceThresholdPolicyChecker,
    DataCollector,
    FixedRatePolicyChecker,
)


class TestFixedRatePolicyCheckerUnit:
    """Unit tests for FixedRatePolicyChecker."""

    def test_should_collect_true(self):
        # Arrange
        policy = FixedRateDataCollectionPolicy(rate=0.1)

        # Act
        should_collect = FixedRatePolicyChecker(policy).should_collect(100, [50.0])

        # Assert
        assert should_collect is True

    def test_should_collect_false(self):
        # Arrange
        policy = FixedRateDataCollectionPolicy(rate=0.1)

        # Act
        should_collect = FixedRatePolicyChecker(policy).should_collect(9, [50.0])

        # Assert
        assert should_collect is False


class TestConfidenceThresholdDataCollectionPolicyUnit:
    """Unit tests for ConfidenceThresholdDataCollectionPolicy."""

    def test_should_collect_true(self):
        # Arrange
        policy = ConfidenceThresholdDataCollectionPolicy(confidence_threshold=30.0, min_sampling_interval=5)

        # Act
        should_collect = ConfidenceThresholdPolicyChecker(policy).should_collect(6, [25.0])

        # Assert
        assert should_collect is True

    def test_should_collect_high_confidence(self):
        # Arrange
        policy = ConfidenceThresholdDataCollectionPolicy(confidence_threshold=30.0, min_sampling_interval=5)

        # Act
        should_collect = ConfidenceThresholdPolicyChecker(policy).should_collect(6, [35.0])

        # Assert
        assert should_collect is False

    def test_should_collect_sampling_interval(self):
        # Arrange
        policy = ConfidenceThresholdDataCollectionPolicy(confidence_threshold=30.0, min_sampling_interval=5)

        # Act
        should_collect = ConfidenceThresholdPolicyChecker(policy).should_collect(3, [25.0])

        # Assert
        assert should_collect is False


@pytest.fixture
def fxt_data_collector(fxt_event_bus) -> DataCollector:
    """Fixture to create a DataCollector instance with mocked dependencies."""
    with patch("app.services.data_collect.data_collector.DataCollector._load_pipeline"):
        return DataCollector(Path("data"), fxt_event_bus)


class TestDataCollectorUnit:
    """Unit tests for DataCollector."""

    def test_collect_no(self, fxt_data_collector):
        """
        No images should be collected if policy conditions aren't met
        """
        # Arrange
        frame_data = np.random.rand(100, 100, 3)
        inference_data = MagicMock()

        now = datetime.timestamp(datetime.now())

        policy_checker = MagicMock()
        policy_checker.should_collect.return_value = False
        fxt_data_collector.policy_checkers = [policy_checker]

        # Act
        with (
            patch.object(DatasetService, "create_dataset_item") as mock_create_dataset_item,
            patch("app.services.data_collect.data_collector.convert_prediction") as mock_convert_prediction,
        ):
            fxt_data_collector.collect(
                timestamp=now + 1,
                frame_data=frame_data,
                inference_data=inference_data,
            )

        # Assert
        mock_convert_prediction.assert_not_called()
        mock_create_dataset_item.assert_not_called()

    @time_machine.travel("2025-01-01 00:00:01 +0000", tick=False)
    def test_collect_by_flag(self, fxt_data_collector):
        """
        Image should be collected if should_collect_next_frame flag has been set
        """
        # Arrange
        pipeline = MagicMock()
        project = MagicMock()
        label = MagicMock(spec=Label)
        frame_data = np.random.randint(low=0, high=255, size=(100, 100), dtype=np.uint8)
        inference_data = MagicMock()

        now = datetime.timestamp(datetime.now())

        policy_checker = MagicMock()
        policy_checker.should_collect.return_value = False
        fxt_data_collector.active_pipeline_data = pipeline, project
        fxt_data_collector.policy_checkers = [policy_checker]
        fxt_data_collector.should_collect_next_frame = True

        annotations = [DatasetItemAnnotation(labels=[LabelReference(id=uuid4())], shape=FullImage())]

        # Act
        with (
            patch.object(DatasetService, "create_dataset_item") as mock_create_dataset_item,
            patch.object(LabelService, "list_all", return_value=[label]) as mock_list_all,
            patch(
                "app.services.data_collect.data_collector.convert_prediction", return_value=annotations
            ) as mock_convert_prediction,
        ):
            fxt_data_collector.collect(
                timestamp=now,
                frame_data=frame_data,
                inference_data=inference_data,
            )

        # Assert
        mock_list_all.assert_called_once_with(project_id=project.id)
        mock_convert_prediction.assert_called_once_with(
            labels=[label], frame_data=ANY, prediction=inference_data.prediction
        )
        mock_create_dataset_item.assert_called_once_with(
            project=project,
            name="1735689601_0000",
            format=DatasetItemFormat.JPG,
            data=ANY,
            user_reviewed=False,
            source_id=pipeline.source_id,
            prediction_model_id=inference_data.model_id,
            annotations=annotations,
        )

    @time_machine.travel("2025-01-01 00:00:01 +0000", tick=False)
    def test_collect(self, fxt_data_collector):
        """
        Image should be collected if policy conditions are met
        """
        # Arrange
        pipeline = MagicMock()
        project = MagicMock()
        label = MagicMock(spec=Label)
        frame_data = np.random.randint(low=0, high=255, size=(100, 100), dtype=np.uint8)
        inference_data = MagicMock()

        now = datetime.timestamp(datetime.now())

        policy_checker = MagicMock()
        policy_checker.should_collect.return_value = True
        fxt_data_collector.active_pipeline_data = pipeline, project
        fxt_data_collector.policy_checkers = [policy_checker]
        fxt_data_collector.should_collect_next_frame = False

        annotations = [DatasetItemAnnotation(labels=[LabelReference(id=uuid4())], shape=FullImage())]

        # Act
        with (
            patch.object(DatasetService, "create_dataset_item") as mock_create_dataset_item,
            patch.object(LabelService, "list_all", return_value=[label]) as mock_list_all,
            patch(
                "app.services.data_collect.data_collector.convert_prediction", return_value=annotations
            ) as mock_convert_prediction,
        ):
            fxt_data_collector.collect(
                timestamp=now,
                frame_data=frame_data,
                inference_data=inference_data,
            )

        # Assert
        mock_list_all.assert_called_once_with(project_id=project.id)
        mock_convert_prediction.assert_called_once_with(
            labels=[label], frame_data=ANY, prediction=inference_data.prediction
        )
        mock_create_dataset_item.assert_called_once_with(
            project=project,
            name="1735689601_0000",
            format=DatasetItemFormat.JPG,
            data=ANY,
            user_reviewed=False,
            source_id=pipeline.source_id,
            prediction_model_id=inference_data.model_id,
            annotations=annotations,
        )
