# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestModelArchitecturesEndpoint:
    """Test cases for the model architectures endpoint."""

    def test_get_all_model_architectures(self, fxt_client: TestClient):
        """Test getting all model architectures without filtering."""
        response = fxt_client.get("/api/model_architectures")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "model_architectures" in data
        assert len(data["model_architectures"]) == 27

        # Verify structure of first detection model
        detection_model = next(
            arch for arch in data["model_architectures"] if arch["id"] == "Custom_Object_Detection_Gen3_ATSS"
        )

        assert detection_model["task"] == "detection"
        assert detection_model["name"] == "ATSS-MobileNetV2"
        assert (
            detection_model["description"]
            == "ATSS (Adaptive Training Sample Selection) is an anchor-based object detection algorithm that introduces"
            " an adaptive strategy for selecting positive and negative samples during training. Instead of using"
            " fixed IoU thresholds, ATSS dynamically determines positive samples based on the statistical"
            " characteristics of object candidates for each ground truth. This improves training stability"
            " and enhances detection performance, especially for objects of varying sizes and aspect ratios."
        )
        assert detection_model["support_status"] == "active"

        # Verify capabilities structure
        assert "capabilities" in detection_model
        assert detection_model["capabilities"]["xai"] is True
        assert detection_model["capabilities"]["tiling"] is True

        # Verify stats structure
        assert "stats" in detection_model
        assert detection_model["stats"]["gigaflops"] == 20.6
        assert detection_model["stats"]["trainable_parameters"] == 3.9
        assert "performance_ratings" in detection_model["stats"]

    @pytest.mark.parametrize(
        "task_filter, total_models",
        [
            ("Detection", 15),
            ("Instance_Segmentation", 5),
            ("Classification", 7),
        ],
    )
    def test_get_model_architectures_various_tasks(self, fxt_client: TestClient, task_filter, total_models):
        """Test getting model architectures with various task filters."""
        response = fxt_client.get(f"/api/model_architectures?task={task_filter}")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "model_architectures" in data
        assert len(data["model_architectures"]) == total_models

        for model in data["model_architectures"]:
            assert model["task"].lower() == task_filter.lower()

    def test_get_model_architectures_nonexistent_task_filter(self, fxt_client: TestClient):
        """Test filtering by a task that doesn't exist returns empty list."""
        response = fxt_client.get("/api/model_architectures?task=nonexistent")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "model_architectures" in data
        assert len(data["model_architectures"]) == 0
