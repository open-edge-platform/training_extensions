import json
import re
import time
from datetime import datetime
from unittest.mock import Mock, patch

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import pytest
from testcontainers.compose import DockerCompose

from app.schemas.configuration import OutputFormat
from app.schemas.configuration.output_config import MqttOutputConfig
from app.services.dispatchers.mqtt import MqttDispatcher


@pytest.fixture(scope="session")
def mqtt_broker():
    """Start MQTT broker using testcontainers."""
    compose = DockerCompose("tests/integration/fixtures", compose_file_name="docker-compose.test.mqtt.yaml")
    compose.start()

    # Wait for MQTT broker to be ready - check logs directly
    mqtt_logs = compose.get_logs("mqtt")

    # Wait for the broker to start
    timeout = 30
    start_time = time.time()
    while time.time() - start_time < timeout:
        if re.search(r"mosquitto version .* starting", mqtt_logs[0] if mqtt_logs else ""):
            break
        time.sleep(1)
    else:
        raise TimeoutError("MQTT broker did not start within timeout")

    time.sleep(2)  # Additional wait for broker to be fully ready

    # Get the exposed port
    mqtt_port = compose.get_service_port("mqtt", 1883)

    yield "localhost", mqtt_port

    compose.stop()


@pytest.fixture
def mqtt_config(mqtt_broker) -> MqttOutputConfig:
    """Create MQTT configuration for testing."""
    host, port = mqtt_broker
    return MqttOutputConfig(
        destination_type="mqtt",
        broker_host=host,
        broker_port=port,
        topic="topic",
        username=None,
        password=None,
        output_formats=[OutputFormat.IMAGE_ORIGINAL, OutputFormat.PREDICTIONS],
    )


@pytest.fixture
def mqtt_config_with_auth(mqtt_broker) -> MqttOutputConfig:
    """Create MQTT configuration with authentication."""
    host, port = mqtt_broker
    return MqttOutputConfig(
        destination_type="mqtt",
        broker_host=host,
        broker_port=port,
        topic="topic",
        username="testuser",
        password="testpass",
        output_formats=[OutputFormat.IMAGE_ORIGINAL, OutputFormat.PREDICTIONS],
    )


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_predictions():
    """Create sample predictions for testing."""
    mock_result = Mock()
    mock_result.__str__ = Mock(return_value="test predictions")
    return mock_result


@pytest.fixture
def mqtt_test_subscriber(mqtt_broker):
    """Create a test MQTT subscriber to verify messages."""
    host, port = mqtt_broker

    class TestSubscriber:
        def __init__(self):
            self.received_messages = []
            self.client = mqtt.Client(client_id="test_subscriber")
            self.client.on_message = self._on_message

        def _on_message(self, client, userdata, msg):
            try:
                payload = json.loads(msg.payload.decode())
                self.received_messages.append({"topic": msg.topic, "payload": payload, "timestamp": datetime.now()})
            except json.JSONDecodeError:
                self.received_messages.append(
                    {"topic": msg.topic, "payload": msg.payload.decode(), "timestamp": datetime.now()}
                )

        def connect_and_subscribe(self, topic):
            self.client.connect(host, port)
            self.client.loop_start()
            self.client.subscribe(topic)
            time.sleep(0.5)  # Wait for subscription

        def disconnect(self):
            self.client.loop_stop()
            self.client.disconnect()

        def wait_for_messages(self, count=1, timeout=5):
            start_time = time.time()
            while len(self.received_messages) < count and time.time() - start_time < timeout:
                time.sleep(0.1)
            return len(self.received_messages) >= count

    subscriber = TestSubscriber()
    yield subscriber
    subscriber.disconnect()


class TestMqttDispatcher:
    """Test cases for MqttDispatcher."""

    def test_init_successful_connection(self, mqtt_config):
        """Test successful initialization and connection."""
        dispatcher = MqttDispatcher(mqtt_config, track_messages=True)

        assert dispatcher.is_connected
        assert dispatcher.broker_host == mqtt_config.broker_host
        assert dispatcher.broker_port == mqtt_config.broker_port
        assert dispatcher.topic == mqtt_config.topic

        dispatcher.close()

    def test_init_with_authentication(self, mqtt_config_with_auth):
        """Test initialization with username/password."""
        dispatcher = MqttDispatcher(mqtt_config_with_auth, track_messages=True)

        assert dispatcher.username == "testuser"
        assert dispatcher.password == "testpass"

        dispatcher.close()

    def test_init_missing_paho_mqtt(self, mqtt_config):
        """Test initialization fails when paho-mqtt is not available."""
        with (
            patch("app.services.dispatchers.mqtt.mqtt", None),
            pytest.raises(ImportError, match="paho-mqtt is required"),
        ):
            MqttDispatcher(mqtt_config)

    def test_connection_failure(self):
        """Test connection failure to invalid broker."""
        config = MqttOutputConfig(
            destination_type="mqtt",
            broker_host="invalid_host",
            broker_port=1883,
            topic="topic",
            output_formats=[OutputFormat.PREDICTIONS],
        )

        with pytest.raises(ConnectionError, match="Failed to connect to MQTT broker"):
            MqttDispatcher(config)

    def test_dispatch_image_original(self, mqtt_config, sample_image, mqtt_test_subscriber):
        """Test dispatching original image."""
        mqtt_test_subscriber.connect_and_subscribe("topic")

        dispatcher = MqttDispatcher(mqtt_config, track_messages=True)
        dispatcher._dispatch_image(sample_image, data_type=OutputFormat.IMAGE_ORIGINAL)

        assert len(dispatcher.get_published_messages()) == 1
        assert mqtt_test_subscriber.wait_for_messages(1)

        message = mqtt_test_subscriber.received_messages[0]
        assert message["topic"] == "topic"
        assert message["payload"]["type"] == OutputFormat.IMAGE_ORIGINAL
        assert "image" in message["payload"]
        assert message["payload"]["format"] == "jpeg"
        assert "timestamp" in message["payload"]

        dispatcher.close()

    def test_dispatch_predictions(self, mqtt_config, sample_predictions, mqtt_test_subscriber):
        """Test dispatching predictions."""
        mqtt_test_subscriber.connect_and_subscribe("topic")

        dispatcher = MqttDispatcher(mqtt_config, track_messages=True)
        dispatcher._dispatch_predictions(sample_predictions)

        assert len(dispatcher.get_published_messages()) == 1
        assert mqtt_test_subscriber.wait_for_messages(1)

        message = mqtt_test_subscriber.received_messages[0]
        assert message["topic"] == "topic"
        assert message["payload"]["type"] == OutputFormat.PREDICTIONS
        assert message["payload"]["predictions"] == "test predictions"

        dispatcher.close()

    def test_full_dispatch_workflow(self, mqtt_config, sample_image, sample_predictions, mqtt_test_subscriber):
        """Test complete dispatch workflow."""
        # Update config to include all formats
        mqtt_config.output_formats = [
            OutputFormat.IMAGE_ORIGINAL,
            OutputFormat.IMAGE_WITH_PREDICTIONS,
            OutputFormat.PREDICTIONS,
        ]

        mqtt_test_subscriber.connect_and_subscribe("topic")

        dispatcher = MqttDispatcher(mqtt_config, track_messages=True)

        viz_image = sample_image.copy()
        cv2.rectangle(viz_image, (10, 10), (50, 50), (0, 255, 0), 2)

        dispatcher._dispatch(sample_image, viz_image, sample_predictions)

        assert len(dispatcher.get_published_messages()) == 3
        assert mqtt_test_subscriber.wait_for_messages(3, timeout=10)

        message_types = [msg["payload"]["type"] for msg in mqtt_test_subscriber.received_messages]
        assert OutputFormat.IMAGE_ORIGINAL in message_types
        assert OutputFormat.IMAGE_WITH_PREDICTIONS in message_types
        assert OutputFormat.PREDICTIONS in message_types

        dispatcher.close()

    def test_reconnection_on_disconnect(self, mqtt_config, sample_image):
        """Test reconnection behavior when disconnected."""
        dispatcher = MqttDispatcher(mqtt_config, track_messages=True)

        # Simulate disconnect
        dispatcher.client.disconnect()
        time.sleep(1)

        # Try to dispatch - should trigger reconnection
        dispatcher._dispatch_image(sample_image, OutputFormat.IMAGE_ORIGINAL)

        # Should reconnect and be connected again
        time.sleep(2)
        assert dispatcher.is_connected

        dispatcher.close()

    def test_publish_failure_handling(self, mqtt_config):
        """Test handling of publish failures."""
        dispatcher = MqttDispatcher(mqtt_config, track_messages=True)

        # Mock client to simulate publish failure
        with patch.object(dispatcher.client, "publish") as mock_publish:
            mock_result = Mock()
            mock_result.rc = mqtt.MQTT_ERR_NO_CONN
            mock_publish.return_value = mock_result

            result = dispatcher._publish_message("test/topic", {"test": "data"})
            assert result is False

        dispatcher.close()

    def test_close_dispatcher(self, mqtt_config):
        """Test proper cleanup when closing dispatcher."""
        dispatcher = MqttDispatcher(mqtt_config)

        assert dispatcher.is_connected

        dispatcher.close()

        assert not dispatcher.is_connected
