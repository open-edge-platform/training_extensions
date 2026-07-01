# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from app.models.inference import InferenceWorkerStatus, InferenceWorkerStatusCode
from app.services.inference_status_service import InferenceStatusService


class TestInferenceStatusService:
    """Test cases for InferenceStatusService.get_status"""

    @pytest.fixture
    def fxt_inference_status_service(self) -> InferenceStatusService:
        return InferenceStatusService(inference_status_shm=MagicMock(), inference_status_lock=MagicMock())

    def test_get_status_matching_model_id(self, fxt_inference_status_service: InferenceStatusService):
        """Return the status when its model_id matches the requested one."""
        model_id = uuid4()
        status = InferenceWorkerStatus(code=InferenceWorkerStatusCode.OK, model_id=model_id, message="inferring")

        with patch("app.services.inference_status_service.read_status", return_value=status) as mock_read:
            result = fxt_inference_status_service.get_status(model_id)

        assert result is status
        mock_read.assert_called_once_with(
            InferenceWorkerStatus,
            fxt_inference_status_service._inference_status_shm,
            fxt_inference_status_service._inference_status_lock,
        )

    def test_get_status_mismatched_model_id(self, fxt_inference_status_service: InferenceStatusService):
        """Return None when the latest status belongs to a different model."""
        status = InferenceWorkerStatus(code=InferenceWorkerStatusCode.OK, model_id=uuid4())

        with patch("app.services.inference_status_service.read_status", return_value=status):
            result = fxt_inference_status_service.get_status(uuid4())

        assert result is None

    def test_get_status_no_status_available(self, fxt_inference_status_service: InferenceStatusService):
        """Return None when no status is available in shared memory."""
        with patch("app.services.inference_status_service.read_status", return_value=None):
            result = fxt_inference_status_service.get_status(uuid4())

        assert result is None
