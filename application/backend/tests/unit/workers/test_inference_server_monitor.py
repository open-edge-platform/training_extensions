# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch
from uuid import uuid4

from app.models import BatchInferenceInput, BatchInferenceResult, Label
from app.workers import InferenceServerMonitorThread


class TestInferenceServerMonitorThread:
    def test_set_inference_model_success(self) -> None:
        project_id = uuid4()
        model_id = uuid4()
        mock_server = MagicMock()
        orig_set_inference_model = mock_server.set_inference_model
        orig_set_inference_model.return_value = True

        monitor_thread = InferenceServerMonitorThread(server=mock_server, stop_event=MagicMock())
        monitor_thread.setup()

        # Simulate loading a model with a TTL
        ttl_value = 60
        returned_result = mock_server.set_inference_model(
            project_id=project_id, model_id=model_id, device="AUTO", ttl=ttl_value
        )

        assert returned_result is True
        assert monitor_thread._ttl == ttl_value
        assert monitor_thread._ttl_start_time > 0
        orig_set_inference_model.assert_called_once_with(
            project_id=project_id, model_id=model_id, device="AUTO", ttl=ttl_value, model_variant_id=None
        )

    def test_set_inference_model_not_loaded(self) -> None:
        project_id = uuid4()
        model_id = uuid4()
        mock_server = MagicMock()
        orig_set_inference_model = mock_server.set_inference_model
        orig_set_inference_model.return_value = False

        monitor_thread = InferenceServerMonitorThread(server=mock_server, stop_event=MagicMock())
        monitor_thread.setup()

        # Simulate loading a model with a TTL
        returned_result = mock_server.set_inference_model(
            project_id=project_id, model_id=model_id, device="AUTO", ttl=60
        )

        assert returned_result is False
        assert monitor_thread._ttl == 0
        assert monitor_thread._ttl_start_time < 0
        orig_set_inference_model.assert_called_once_with(
            project_id=project_id, model_id=model_id, device="AUTO", ttl=60, model_variant_id=None
        )

    def test_infer_batch(self) -> None:
        label = MagicMock(spec=Label)
        input = MagicMock(spec=BatchInferenceInput)
        result = MagicMock(spec=BatchInferenceResult)
        mock_server = MagicMock()
        orig_infer_batch = mock_server.infer_batch
        orig_infer_batch.return_value = result

        monitor_thread = InferenceServerMonitorThread(server=mock_server, stop_event=MagicMock())
        monitor_thread.setup()

        # Simulate inference request
        returned_result = mock_server.infer_batch(labels=[label], inputs=[input])

        assert returned_result == result
        assert monitor_thread._ttl_start_time > 0
        orig_infer_batch.assert_called_once_with(labels=[label], inputs=[input])

    def test_stop(self) -> None:
        mock_server = MagicMock()
        orig_stop = mock_server.stop

        monitor_thread = InferenceServerMonitorThread(server=mock_server, stop_event=MagicMock())
        monitor_thread.setup()

        # Simulate stop request
        mock_server.stop()

        assert monitor_thread._ttl_start_time < 0
        orig_stop.assert_called_once_with()

    def test_run_loop_ttl_expired(self) -> None:
        mock_server = MagicMock()
        stop_method = mock_server.stop

        stop_event = MagicMock()
        stop_event.is_set.side_effect = [False, True]  # Run loop once then stop
        monitor_thread = InferenceServerMonitorThread(server=mock_server, stop_event=stop_event)
        monitor_thread.setup()
        monitor_thread._ttl = 1
        monitor_thread._ttl_start_time = 1

        with patch("time.perf_counter", return_value=100):
            monitor_thread.run_loop()

        assert monitor_thread._ttl_start_time < 0  # Check that TTL countdown is reset
        stop_method.assert_called_once_with()  # Check that server stop was called on TTL expiration

    def test_run_loop_ttl_not_expired(self) -> None:
        mock_server = MagicMock()
        stop_method = mock_server.stop

        stop_event = MagicMock()
        stop_event.is_set.side_effect = [False, True]  # Run loop once then stop
        monitor_thread = InferenceServerMonitorThread(server=mock_server, stop_event=stop_event)
        monitor_thread.setup()
        monitor_thread._ttl = 1000
        monitor_thread._ttl_start_time = 1

        with patch("time.perf_counter", return_value=100):
            monitor_thread.run_loop()

        assert monitor_thread._ttl_start_time > 0
        stop_method.assert_not_called()
