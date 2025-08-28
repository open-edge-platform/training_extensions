# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import numpy as np
import pytest
import requests
from freezegun import freeze_time

from app.schemas import OutputFormat, SinkType
from app.schemas.sink import WebhookSinkConfig
from app.services.dispatchers.base import DispatchError, numpy_to_base64
from app.services.dispatchers.webhook import WebhookDispatcher


@pytest.fixture
def fxt_webhook_config():
    return WebhookSinkConfig(
        sink_type=SinkType.WEBHOOK,
        name="Test Webhook Sink",
        rate_limit=0.2,
        output_formats=[
            OutputFormat.IMAGE_ORIGINAL,
            OutputFormat.IMAGE_WITH_PREDICTIONS,
            OutputFormat.PREDICTIONS,
        ],
        webhook_url="https://example.com/webhook",
        http_method="PATCH",
        headers={"Authorization": "Bearer token"},
        timeout=5,
    )


@pytest.fixture
def fxt_sample_image():
    """Create a sample image for testing."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def fxt_sample_predictions():
    """Create sample predictions for testing."""
    mock_result = Mock()
    mock_result.__str__ = Mock(return_value="test predictions")
    return mock_result


class TestWebhookDispatcher:
    """Unit tests for the WebhookDispatcher class."""

    def test_init_sets_attributes(self, fxt_webhook_config):
        """Test that the WebhookDispatcher initializes with correct attributes."""
        dispatcher = WebhookDispatcher(fxt_webhook_config)

        assert dispatcher.webhook_url == fxt_webhook_config.webhook_url
        assert dispatcher.http_method == fxt_webhook_config.http_method
        assert dispatcher.headers == fxt_webhook_config.headers
        assert dispatcher.timeout == fxt_webhook_config.timeout
        assert dispatcher.session is not None

    @pytest.mark.parametrize(
        "output_formats",
        [
            ([OutputFormat.PREDICTIONS],),
            ([OutputFormat.IMAGE_ORIGINAL],),
            ([OutputFormat.IMAGE_WITH_PREDICTIONS],),
            (
                [
                    OutputFormat.PREDICTIONS,
                    OutputFormat.IMAGE_ORIGINAL,
                    OutputFormat.IMAGE_WITH_PREDICTIONS,
                ],
            ),
        ],
    )
    @freeze_time("2025-01-01T12:00:00")
    def test_dispatch_sends_payload(self, output_formats, fxt_webhook_config, fxt_sample_image, fxt_sample_predictions):
        """Test that the _dispatch method sends the correct payload to the webhook."""
        fxt_webhook_config.output_formats = output_formats
        dispatcher = WebhookDispatcher(fxt_webhook_config)
        with patch.object(dispatcher.session, "request") as mock_request:
            dispatcher.dispatch(
                original_image=fxt_sample_image,
                image_with_visualization=fxt_sample_image,
                predictions=fxt_sample_predictions,
            )

            expected_result = {}
            if OutputFormat.PREDICTIONS in output_formats:
                expected_result[OutputFormat.PREDICTIONS] = "test predictions"
            if OutputFormat.IMAGE_ORIGINAL in output_formats:
                expected_result[OutputFormat.IMAGE_ORIGINAL] = numpy_to_base64(fxt_sample_image)
            if OutputFormat.IMAGE_WITH_PREDICTIONS in output_formats:
                expected_result[OutputFormat.IMAGE_WITH_PREDICTIONS] = numpy_to_base64(fxt_sample_image)

            mock_request.assert_called_once_with(
                fxt_webhook_config.http_method,
                fxt_webhook_config.webhook_url,
                headers=fxt_webhook_config.headers,
                json={
                    "timestamp": "2025-01-01T12:00:00",
                    "result": expected_result,
                },
                timeout=fxt_webhook_config.timeout,
            )

    def test_send_to_webhook_raises_http_error(self, fxt_webhook_config, fxt_sample_image, fxt_sample_predictions):
        dispatcher = WebhookDispatcher(fxt_webhook_config)
        with patch.object(dispatcher.session, "request") as mock_request:
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = requests.HTTPError("Bad Request")
            mock_request.return_value = mock_response

            with pytest.raises(DispatchError):
                dispatcher.dispatch(
                    original_image=fxt_sample_image,
                    image_with_visualization=fxt_sample_image,
                    predictions=fxt_sample_predictions,
                )
