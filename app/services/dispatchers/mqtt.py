import base64
import json
import logging
import threading
import time
from datetime import datetime
from typing import Any

import cv2
import numpy as np
from model_api.models.result import Result

from app.schemas.configuration import OutputFormat
from app.schemas.configuration.output_config import MqttOutputConfig
from app.services.dispatchers.base import BaseDispatcher

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None


logger = logging.getLogger(__name__)
MAX_RETRIES = 3
RETRY_DELAY = 1
CONNECT_TIMEOUT = 10


def _encode_image_to_base64(image: np.ndarray, fmt: str = ".jpg") -> str:
    success, img_buf = cv2.imencode(fmt, image)
    if success:
        return base64.b64encode(img_buf.tobytes()).decode("utf-8")
    raise ValueError(f"Failed to encode image in format {fmt}")


def _create_mqtt_payload(data_type: str, **kwargs) -> dict[str, Any]:
    return {"timestamp": datetime.now().isoformat(), "type": data_type, **kwargs}


class MqttDispatcher(BaseDispatcher):
    def __init__(
        self,
        output_config: MqttOutputConfig,
        mqtt_client: "mqtt.Client | None" = None,
        track_messages: bool | None = False,
    ) -> None:
        """
        Initialize the MqttDispatcher.

        Args:
            output_config: Configuration for the MQTT destination
            mqtt_client: MQTT client
            track_messages: Flag to track MQTT messages (useful for debugging/testing)

        Raises:
            ImportError: If paho-mqtt is not installed
            ConnectionError: If unable to connect to MQTT broker
        """
        if mqtt is None:
            raise ImportError("paho-mqtt is required for MQTT dispatcher.")

        super().__init__(output_config)
        self.broker_host = output_config.broker_host
        self.broker_port = output_config.broker_port
        self.topic = output_config.topic
        self.username = output_config.username
        self.password = output_config.password

        self._connected = False
        self._connection_lock = threading.Lock()
        self._connection_event = threading.Event()
        self._track_messages = track_messages
        self._published_messages: list[dict] = []

        self.client = mqtt_client or self._create_default_client()
        self._connect()

    def _create_default_client(self) -> "mqtt.Client":
        client_id = f"dispatcher_{int(time.time())}"
        client = mqtt.Client(client_id=client_id)
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        if self.username and self.password:
            client.username_pw_set(self.username, self.password)
        return client

    def _connect(self) -> None:
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(
                    "Connecting to MQTT broker at %s:%s (attempt %s)", self.broker_host, self.broker_port, attempt + 1
                )
                self.client.connect(self.broker_host, self.broker_port)
                self.client.loop_start()
                if self._connection_event.wait(CONNECT_TIMEOUT):
                    return
                logger.warning("Connection timeout after %s seconds", CONNECT_TIMEOUT)
            except Exception as e:
                logger.exception("Connection failed %s", e)
                time.sleep(RETRY_DELAY * (attempt + 1))
        raise ConnectionError("Failed to connect to MQTT broker")

    def _on_connect(self, _client: "mqtt.Client", _userdata: Any, _flags: dict[str, int], rc: int):
        if rc == 0:
            self._connected = True
            self._connection_event.set()
            logger.info("Connected to MQTT broker")
        else:
            logger.error("MQTT connect failed with code %s", rc)

    def _on_disconnect(self, _client: "mqtt.Client", _userdata: Any, rc: int):
        self._connected = False
        self._connection_event.clear()
        logger.warning("MQTT disconnected (rc=%s)", rc)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _publish_message(self, topic: str, payload: dict[str, Any]) -> bool:
        if not self._connected:
            logger.warning("Client not connected. Reconnecting...")
            try:
                self._connect()
            except Exception:
                logger.exception("Reconnect failed")
                return False

        try:
            result = self.client.publish(topic, json.dumps(payload))
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                if self._track_messages:
                    self._published_messages.append({"topic": topic, "payload": payload, "timestamp": datetime.now()})
                return True
            logger.error(f"Publish failed: {mqtt.error_string(result.rc)}")
        except Exception:
            logger.exception("Publish exception")
        return False

    def _dispatch_image(self, image: np.ndarray, data_type: str):
        try:
            image_b64 = _encode_image_to_base64(image)
            payload = _create_mqtt_payload(
                data_type=data_type,
                image=image_b64,
                format="jpeg",
            )
            self._publish_message(self.topic, payload)
        except Exception:
            logger.exception("Failed to dispatch %s", data_type)

    def _dispatch_predictions(self, predictions: Result):
        try:
            payload = _create_mqtt_payload(data_type=OutputFormat.PREDICTIONS.value, predictions=str(predictions))
            self._publish_message(self.topic, payload)
        except Exception:
            logger.exception("Failed to dispatch predictions")

    def _dispatch(self, original_image: np.ndarray, image_with_visualization: np.ndarray, predictions: Result) -> None:
        if OutputFormat.IMAGE_ORIGINAL in self.output_formats:
            self._dispatch_image(original_image, OutputFormat.IMAGE_ORIGINAL)

        if OutputFormat.IMAGE_WITH_PREDICTIONS in self.output_formats:
            self._dispatch_image(image_with_visualization, OutputFormat.IMAGE_WITH_PREDICTIONS)

        if OutputFormat.PREDICTIONS in self.output_formats:
            self._dispatch_predictions(predictions)

    def get_published_messages(self) -> list:
        return self._published_messages.copy()

    def clear_published_messages(self) -> None:
        self._published_messages.clear()

    def close(self) -> None:
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            logger.exception("Error closing dispatcher")
        finally:
            self._connected = False
            self._connection_event.clear()
