# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from loguru import logger

from app.stream.stream_data import StreamData
from app.workers.inference import InferenceWorker, InferenceWorkerConfig, PredictionReorderBuffer


@pytest.fixture(autouse=True)
def fxt_loguru_caplog(caplog):
    class PropagateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    handler_id = logger.add(PropagateHandler(), format="{message}")
    yield
    logger.remove(handler_id)


@pytest.fixture
def fxt_create_stream_data():
    def create_sample(ts):
        return StreamData(
            frame_data=np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8),
            timestamp=ts,
            source_metadata={},
        )

    return create_sample


@pytest.fixture
def fxt_inference_worker():
    """Construct an InferenceWorker with mocked services, bypassing setup()."""
    config = InferenceWorkerConfig(
        frame_queue=MagicMock(),
        pred_queue=MagicMock(),
        stop_event=MagicMock(),
        model_reload_event=MagicMock(),
        shm_name="shm",
        shm_lock=MagicMock(),
        logger_=logger,  # type: ignore[arg-type]
    )
    worker = InferenceWorker(config)
    worker._model_service = MagicMock()
    worker._metrics_service = MagicMock()
    # Initialize the name-mangled prediction buffer attribute that setup() would normally create.
    setattr(worker, "_InferenceWorker__prediction_buffer", PredictionReorderBuffer())
    return worker


class TestPredictionReorderBuffer:
    def test_register_timestamp_and_add_prediction(self, fxt_create_stream_data):
        buffer = PredictionReorderBuffer(max_size=5)
        ts1, ts2, ts3, ts4, ts5 = 1.0, 2.0, 3.0, 4.0, 5.0
        sd1 = fxt_create_stream_data(ts1)
        sd2 = fxt_create_stream_data(ts2)
        sd3 = fxt_create_stream_data(ts3)
        sd5 = fxt_create_stream_data(ts5)

        # Register timestamps in order 1-5, add stream data out of order and with gaps
        for ts in [ts1, ts2, ts3, ts4, ts5]:
            buffer.register_expected_timestamp(ts)
        buffer.add_prediction_for_timestamp(ts2, sd2)
        buffer.add_prediction_for_timestamp(ts1, sd1)
        buffer.add_prediction_for_timestamp(ts3, sd3)
        buffer.add_prediction_for_timestamp(ts5, sd5)

        # Buffer should output predictions in registration order and without gaps
        ready = buffer.get_ready_predictions()
        assert ready == [sd1, sd2, sd3]

    def test_full_register(self, fxt_create_stream_data):
        # Buffer should only hold 3 expected timestamps
        buffer = PredictionReorderBuffer(max_size=3)

        # Prepare timstamps and stream data
        ts1, ts2, ts3, ts4, ts5 = 1.0, 2.0, 3.0, 4.0, 5.0
        sd1 = fxt_create_stream_data(ts1)
        sd2 = fxt_create_stream_data(ts2)
        sd3 = fxt_create_stream_data(ts3)
        sd4 = fxt_create_stream_data(ts4)
        sd5 = fxt_create_stream_data(ts5)

        # Register first three and add stream data for ts1 and ts3
        for ts in [ts1, ts2, ts3]:
            buffer.register_expected_timestamp(ts)
        buffer.add_prediction_for_timestamp(ts1, sd1)
        buffer.add_prediction_for_timestamp(ts3, sd3)

        # Register last two, first two should be dropped, including stream data
        buffer.register_expected_timestamp(ts4)
        buffer.register_expected_timestamp(ts5)

        # Add stream data for ts2, ts4, ts5. Stream data for ts2 should be dropped
        buffer.add_prediction_for_timestamp(ts2, sd2)
        buffer.add_prediction_for_timestamp(ts5, sd5)
        buffer.add_prediction_for_timestamp(ts4, sd4)

        assert buffer.get_ready_predictions() == [sd3, sd4, sd5]

    def test_add_unexpected_prediction_warns(self, fxt_create_stream_data, fxt_loguru_caplog, caplog):
        buffer = PredictionReorderBuffer(max_size=3)
        ts = 42.0
        sd = fxt_create_stream_data(ts)
        with caplog.at_level("WARNING"):
            buffer.add_prediction_for_timestamp(ts, sd)
        assert "unexpected timestamp" in caplog.text
        assert buffer.get_ready_predictions() == []

    def test_clear(self, fxt_create_stream_data):
        buffer = PredictionReorderBuffer(max_size=2)
        ts = 1.0
        buffer.register_expected_timestamp(ts)
        buffer.add_prediction_for_timestamp(ts, fxt_create_stream_data(ts))
        buffer.clear()
        assert buffer.get_ready_predictions() == []


class TestInferenceWorkerRefreshModel:
    """Tests for _refresh_loaded_model."""

    def test_no_reload_returns_current_model(self, fxt_inference_worker):
        worker = fxt_inference_worker
        worker._model_reload_event.is_set.return_value = False
        loaded = MagicMock()
        worker._model_service.get_loaded_inference_model.return_value = loaded

        result = worker._refresh_loaded_model()

        assert result is loaded

    def test_no_reload_returns_none_when_no_model(self, fxt_inference_worker):
        worker = fxt_inference_worker
        worker._model_reload_event.is_set.return_value = False
        worker._model_service.get_loaded_inference_model.return_value = None

        result = worker._refresh_loaded_model()

        assert result is None

    def test_reload_force_reloads_model(self, fxt_inference_worker):
        worker = fxt_inference_worker
        worker._model_reload_event.is_set.side_effect = [True, True, False]
        loaded = MagicMock()
        worker._model_service.get_loaded_inference_model.return_value = loaded

        result = worker._refresh_loaded_model()

        assert result is loaded
        worker._model_reload_event.clear.assert_called()
        worker._model_service.get_loaded_inference_model.assert_called_once_with(force_reload=True)

    def test_reload_with_failed_load_returns_none(self, fxt_inference_worker):
        worker = fxt_inference_worker
        worker._model_reload_event.is_set.side_effect = [True, True, False]
        worker._model_service.get_loaded_inference_model.return_value = None

        result = worker._refresh_loaded_model()

        assert result is None

    def test_reload_loop_clears_event_until_stable(self, fxt_inference_worker):
        worker = fxt_inference_worker
        worker._model_reload_event.is_set.side_effect = [True, True, True, False]
        worker._model_service.get_loaded_inference_model.return_value = MagicMock()

        worker._refresh_loaded_model()

        assert worker._model_service.get_loaded_inference_model.call_count == 2

    def test_on_inference_completed_calls_visualizer(self, fxt_inference_worker):
        worker = fxt_inference_worker

        stream_data = StreamData(
            frame_data=np.zeros((4, 4, 3), dtype=np.uint8),
            timestamp=1.0,
            source_metadata={},
        )
        worker._prediction_buffer.register_expected_timestamp(1.0)

        with patch("app.workers.inference.Visualizer.overlay_predictions") as mock_overlay:
            mock_overlay.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
            inf_result = MagicMock()
            worker._on_inference_completed(
                inf_result,
                {"inference_start_time": 1.0, "model_id": "mid", "stream_data": stream_data},
            )

        mock_overlay.assert_called_once()
        _, kwargs = mock_overlay.call_args
        assert kwargs["predictions"] is inf_result
