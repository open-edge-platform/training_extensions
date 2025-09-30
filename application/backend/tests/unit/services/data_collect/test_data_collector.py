# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
from freezegun import freeze_time

from app.schemas.dataset_item import DatasetItemAnnotation, DatasetItemFormat
from app.schemas.label import LabelReference
from app.schemas.pipeline import FixedRateDataCollectionPolicy
from app.schemas.shape import FullImage
from app.services import DatasetService
from app.services.data_collect.data_collector import DataCollector, FixedRatePolicyChecker


class TestFixedRatePolicyCheckerUnit:
    """Unit tests for FixedRatePolicyChecker."""

    def test_should_collect_true(self):
        # Arrange
        policy = FixedRateDataCollectionPolicy(rate=0.1)

        # Act
        should_collect = FixedRatePolicyChecker(policy).should_collect(100)

        # Assert
        assert should_collect is True

    def test_should_collect_false(self):
        # Arrange
        policy = FixedRateDataCollectionPolicy(rate=0.1)

        # Act
        should_collect = FixedRatePolicyChecker(policy).should_collect(9)

        # Assert
        assert should_collect is False


@pytest.fixture
def fxt_data_collector(fxt_active_pipeline_service) -> DataCollector:
    """Fixture to create a DataCollector instance with mocked dependencies."""
    return DataCollector(fxt_active_pipeline_service)


class TestDataCollectorUnit:
    """Unit tests for DataCollector."""

    def test_collect_no(self, fxt_data_collector):
        """
        No images should be collected if policy conditions aren't met
        """
        # Arrange
        source_id = uuid4()
        project = MagicMock()
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
                source_id=source_id,
                project=project,
                timestamp=now + 1,
                frame_data=frame_data,
                inference_data=inference_data,
            )

        # Assert
        mock_convert_prediction.assert_not_called()
        mock_create_dataset_item.assert_not_called()

    @freeze_time("2025-01-01 00:00:01")
    def test_collect_by_flag(self, fxt_data_collector):
        """
        Image should be collected if should_collect_next_frame flag has been set
        """
        # Arrange
        source_id = uuid4()
        project = MagicMock()
        frame_data = np.random.rand(100, 100, 3)
        inference_data = MagicMock()

        now = datetime.timestamp(datetime.now())

        policy_checker = MagicMock()
        policy_checker.should_collect.return_value = False
        fxt_data_collector.policy_checkers = [policy_checker]
        fxt_data_collector.should_collect_next_frame = True

        annotations = [DatasetItemAnnotation(labels=[LabelReference(id=uuid4())], shape=FullImage())]

        # Act
        with (
            patch.object(DatasetService, "create_dataset_item") as mock_create_dataset_item,
            patch(
                "app.services.data_collect.data_collector.convert_prediction", return_value=annotations
            ) as mock_convert_prediction,
        ):
            fxt_data_collector.collect(
                source_id=source_id,
                project=project,
                timestamp=now,
                frame_data=frame_data,
                inference_data=inference_data,
            )

        # Assert
        mock_convert_prediction.assert_called_once_with(
            labels=project.task.labels, frame_data=frame_data, prediction=inference_data.prediction
        )
        mock_create_dataset_item.assert_called_once_with(
            project_id=project.id,
            name="1735689601_0000",
            format=DatasetItemFormat.JPG,
            data=frame_data,
            user_reviewed=False,
            source_id=source_id,
            prediction_model_id=inference_data.model_id,
            annotations=annotations,
        )

    @freeze_time("2025-01-01 00:00:01")
    def test_collect(self, fxt_data_collector):
        """
        Image should be collected if policy conditions are met
        """
        # Arrange
        source_id = uuid4()
        project = MagicMock()
        frame_data = np.random.rand(100, 100, 3)
        inference_data = MagicMock()

        now = datetime.timestamp(datetime.now())

        policy_checker = MagicMock()
        policy_checker.should_collect.return_value = True
        fxt_data_collector.policy_checkers = [policy_checker]
        fxt_data_collector.should_collect_next_frame = False

        annotations = [DatasetItemAnnotation(labels=[LabelReference(id=uuid4())], shape=FullImage())]

        # Act
        with (
            patch.object(DatasetService, "create_dataset_item") as mock_create_dataset_item,
            patch(
                "app.services.data_collect.data_collector.convert_prediction", return_value=annotations
            ) as mock_convert_prediction,
        ):
            fxt_data_collector.collect(
                source_id=source_id,
                project=project,
                timestamp=now,
                frame_data=frame_data,
                inference_data=inference_data,
            )

        # Assert
        mock_convert_prediction.assert_called_once_with(
            labels=project.task.labels, frame_data=frame_data, prediction=inference_data.prediction
        )
        mock_create_dataset_item.assert_called_once_with(
            project_id=project.id,
            name="1735689601_0000",
            format=DatasetItemFormat.JPG,
            data=frame_data,
            user_reviewed=False,
            source_id=source_id,
            prediction_model_id=inference_data.model_id,
            annotations=annotations,
        )
