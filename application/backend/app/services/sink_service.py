# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import socket
import time
from uuid import UUID

import requests
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.schema import SinkDB
from app.models import OutputFormat, Sink, SinkAdapter, SinkType
from app.models.sink import FolderSinkConfig, MqttSinkConfig, SinkConfig, WebhookSinkConfig
from app.repositories import SinkRepository
from app.repositories.base import PrimaryKeyIntegrityError, UniqueConstraintIntegrityError

from .base import (
    ResourceInUseError,
    ResourceNotFoundError,
    ResourceType,
    ResourceWithIdAlreadyExistsError,
    ResourceWithNameAlreadyExistsError,
)
from .event.event_bus import EventBus, EventType
from .parent_process_guard import parent_process_only


class SinkService:
    def __init__(self, event_bus: EventBus, db_session: Session):
        self._event_bus: EventBus = event_bus
        self._db_session = db_session

    @parent_process_only
    def create_sink(
        self,
        name: str,
        sink_type: SinkType,
        rate_limit: float | None,
        config_data: SinkConfig,
        output_formats: list[OutputFormat],
        sink_id: UUID | None = None,
    ) -> Sink:
        try:
            db_sink = SinkRepository(self._db_session).save(
                SinkDB(
                    id=str(sink_id) if sink_id is not None else None,
                    name=name,
                    sink_type=sink_type,
                    rate_limit=rate_limit,
                    config_data=config_data.model_dump(mode="json"),
                    output_formats=output_formats,
                )
            )
            return SinkAdapter.validate_python(db_sink, from_attributes=True)
        except PrimaryKeyIntegrityError:
            raise ResourceWithIdAlreadyExistsError(ResourceType.SINK, str(sink_id))
        except UniqueConstraintIntegrityError:
            raise ResourceWithNameAlreadyExistsError(ResourceType.SINK, name)

    @parent_process_only
    def update_sink(
        self,
        sink: Sink,
        new_name: str,
        new_rate_limit: float | None,
        new_config_data: SinkConfig,
        new_output_formats: list[OutputFormat],
    ) -> Sink:
        try:
            sink_repo = SinkRepository(self._db_session)
            db_sink = sink_repo.update(
                SinkDB(
                    id=str(sink.id),
                    name=new_name,
                    rate_limit=new_rate_limit,
                    config_data=new_config_data.model_dump(mode="json"),
                    output_formats=new_output_formats,
                )
            )
            active_sink_id = self.get_active_sink_id()
            if active_sink_id == UUID(db_sink.id):
                self._event_bus.emit_event(EventType.SINK_CHANGED)
            return SinkAdapter.validate_python(db_sink, from_attributes=True)
        except UniqueConstraintIntegrityError:
            raise ResourceWithNameAlreadyExistsError(ResourceType.SINK, new_name)  # type: ignore[arg-type]

    def get_by_id(self, sink_id: UUID) -> Sink:
        db_sink = SinkRepository(self._db_session).get_by_id(str(sink_id))
        if not db_sink:
            raise ResourceNotFoundError(ResourceType.SINK, str(sink_id))
        return SinkAdapter.validate_python(db_sink, from_attributes=True)

    def list_all(self) -> list[Sink]:
        return [
            SinkAdapter.validate_python(db_sink, from_attributes=True)
            for db_sink in SinkRepository(self._db_session).list_all()
        ]

    @parent_process_only
    def delete_sink(self, sink: Sink) -> None:
        try:
            deleted = SinkRepository(self._db_session).delete(str(sink.id))
            if not deleted:
                raise ResourceNotFoundError(ResourceType.SINK, str(sink.id))
        except IntegrityError:
            raise ResourceInUseError(ResourceType.SINK, str(sink.id))

    def get_active_sink(self) -> Sink | None:
        db_sink = SinkRepository(self._db_session).get_active_sink()
        return SinkAdapter.validate_python(db_sink, from_attributes=True) if db_sink else None

    def get_active_sink_id(self) -> UUID | None:
        id = SinkRepository(self._db_session).get_active_sink_id()
        return UUID(id) if id else None

    _TEST_TIMEOUT_SECONDS = 5

    def test_sink(self, sink: Sink) -> dict:
        """Perform a connectivity check on the sink.

        Verifies that the sink can be reached based on its type:
        - Folder: verifies the folder path exists and is writable
        - MQTT: attempts a TCP connection to the broker host and port
        - Webhook: sends a HEAD request to the webhook URL
        """
        start = time.monotonic()

        try:
            match sink:
                case FolderSinkConfig():
                    reachable, error = self._test_folder(sink)
                case MqttSinkConfig():
                    reachable, error = self._test_mqtt(sink)
                case WebhookSinkConfig():
                    reachable, error = self._test_webhook(sink)
                case _:
                    return {"reachable": False, "error": f"Unsupported sink type: {sink.sink_type}"}
        except Exception as e:
            return {"reachable": False, "error": str(e)}

        if not reachable:
            return {"reachable": False, "error": error}

        elapsed_ms = (time.monotonic() - start) * 1000
        return {"reachable": True, "latency_ms": round(elapsed_ms, 1)}

    def _test_folder(self, sink: FolderSinkConfig) -> tuple[bool, str | None]:
        folder_path = sink.config_data.folder_path
        if not os.path.isdir(folder_path):
            return False, f"Directory not found: {folder_path}"
        if not os.access(folder_path, os.W_OK):
            return False, f"Directory is not writable: {folder_path}"
        return True, None

    def _test_mqtt(self, sink: MqttSinkConfig) -> tuple[bool, str | None]:
        host = sink.config_data.broker_host
        port = sink.config_data.broker_port
        try:
            sock = socket.create_connection((host, port), timeout=self._TEST_TIMEOUT_SECONDS)
            sock.close()
        except OSError as e:
            return False, f"Cannot connect to MQTT broker at {host}:{port}: {e}"
        return True, None

    def _test_webhook(self, sink: WebhookSinkConfig) -> tuple[bool, str | None]:
        url = sink.config_data.webhook_url
        headers = sink.config_data.headers or {}
        try:
            response = requests.head(url, headers=headers, timeout=self._TEST_TIMEOUT_SECONDS)
            if response.status_code >= 500:
                return False, f"Webhook at {url} returned server error: {response.status_code}"
        except requests.RequestException as e:
            return False, f"Cannot reach webhook at {url}: {e}"
        return True, None
