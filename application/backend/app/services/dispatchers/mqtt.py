# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger
from model_api.models.result import Result

from app.models import MqttSinkConfig

from .base import BaseDispatcher

if TYPE_CHECKING:
    try:
        import paho.mqtt.client as mqtt_cl
    except ImportError:
        raise ImportError("Package 'paho-mqtt' is required for type checking. Please install it through extra 'mqtt'.")

MAX_RETRIES = 3
RETRY_DELAY = 1
CONNECT_TIMEOUT = 10


class MqttDispatcher(BaseDispatcher):
    def __init__(
        self,
        output_config: MqttSinkConfig,
        mqtt_client: "mqtt_cl.Client | None" = None,
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
        try:
            import paho.mqtt.client
            import paho.mqtt.enums

            self.mqtt_cl = paho.mqtt.client
            self.mqtt_enums = paho.mqtt.enums
        except ImportError:
            raise ImportError("Package 'paho-mqtt' is required for MQTT dispatcher. Please install with extra 'mqtt'.")

        super().__init__(output_config)
        self.broker_host = output_config.config_data.broker_host
        self.broker_port = output_config.config_data.broker_port
        self.topic = output_config.config_data.topic
        self.username, self.password = output_config.config_data.get_credentials()

        self._connected = False
        self._connection_lock = threading.Lock()
        self._connection_event = threading.Event()
        self._track_messages = track_messages
        self._published_messages: list[dict] = []

        self.client = mqtt_client or self._create_default_client()
        self._connect()

    def _create_default_client(self) -> "mqtt_cl.Client":
        client_id = f"dispatcher_{int(time.time())}"
        client = self.mqtt_cl.Client(
            callback_api_version=self.mqtt_enums.CallbackAPIVersion.VERSION2, client_id=client_id
        )
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        if self.username is not None and self.password is not None:
            client.username_pw_set(self.username, self.password)
        return client

    def _connect(self) -> None:
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(
                    "Connecting to MQTT broker at {}:{} (attempt {})", self.broker_host, self.broker_port, attempt + 1
                )
                self.client.connect(self.broker_host, self.broker_port)
                self.client.loop_start()
                if self._connection_event.wait(CONNECT_TIMEOUT):
                    return
                logger.warning("Connection timeout after {} seconds", CONNECT_TIMEOUT)
            except Exception:
                logger.exception("Connection failed")
                time.sleep(RETRY_DELAY * (attempt + 1))
        raise ConnectionError("Failed to connect to MQTT broker")

    def _on_connect(
        self,
        _client: "mqtt_cl.Client",
        _userdata: Any,
        _flags: "mqtt_cl.ConnectFlags",
        rc: "mqtt_cl.ReasonCode",
        _properties: "mqtt_cl.Properties | None",
    ):
        if rc == 0:
            self._connected = True
            self._connection_event.set()
            logger.info("Connected to MQTT broker")
        else:
            logger.error("MQTT connect failed with code {}", rc)

    def _on_disconnect(
        self,
        _client: "mqtt_cl.Client",
        _userdata: Any,
        _flags: "mqtt_cl.DisconnectFlags",
        rc: "mqtt_cl.ReasonCode",
        _properties: "mqtt_cl.Properties | None",
    ):
        self._connected = False
        self._connection_event.clear()
        logger.warning("MQTT disconnected (rc={})", rc)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def __publish_message(self, topic: str, payload: dict[str, Any]) -> None:
        if not self._connected:
            logger.warning("Client not connected. Reconnecting...")
            try:
                self._connect()
            except ConnectionError:
                logger.exception("Reconnect failed")

        try:
            result = self.client.publish(topic, json.dumps(payload))
            if result.rc == self.mqtt_cl.MQTT_ERR_SUCCESS and self._track_messages:
                self._published_messages.append({"topic": topic, "payload": payload})
            if result.rc != self.mqtt_cl.MQTT_ERR_SUCCESS:
                logger.error("Publish failed: {}", self.mqtt_cl.error_string(result.rc))
        except ValueError:
            logger.exception("Invalid payload for MQTT publish")

    def _dispatch(self, original_image: np.ndarray, image_with_visualization: np.ndarray, predictions: Result) -> None:
        payload = self._create_payload(original_image, image_with_visualization, predictions)

        self.__publish_message(self.topic, payload)

    def get_published_messages(self) -> list:
        return self._published_messages.copy()

    def clear_published_messages(self) -> None:
        self._published_messages.clear()

    def close(self) -> None:
        err = self.client.loop_stop()
        if err != self.mqtt_cl.MQTT_ERR_SUCCESS:
            logger.warning("Error stopping MQTT loop: {}", self.mqtt_cl.error_string(err))
        err = self.client.disconnect()
        if err != self.mqtt_cl.MQTT_ERR_SUCCESS:
            logger.warning("Error disconnecting MQTT client: {}", self.mqtt_cl.error_string(err))
        self._connected = False
        self._connection_event.clear()
