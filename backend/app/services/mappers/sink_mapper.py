# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from app.db.schema import SinkDB
from app.schemas.sink import Sink, SinkAdapter, SinkType


class SinkMapper:
    """Mapper for Sink model <-> Sink schema conversions."""

    # Define fields to exclude from config_data (common fields)
    _COMMON_FIELDS: set[str] = {"id", "name", "sink_type", "output_formats", "rate_limit", "created_at", "updated_at"}

    @staticmethod
    def to_schema(sink_db: SinkDB) -> Sink:
        """Convert Sink model to Sink schema."""

        config_data = sink_db.config_data or {}
        return SinkAdapter.validate_python(
            {
                "id": sink_db.id,
                "name": sink_db.name,
                "sink_type": SinkType(sink_db.sink_type),
                "output_formats": sink_db.output_formats,
                "rate_limit": sink_db.rate_limit,
                **config_data,
            }
        )

    @staticmethod
    def from_schema(sink: Sink) -> SinkDB:
        """Convert Sink schema to Sink model."""
        if sink is None:
            raise ValueError("Sink config cannot be None")

        sink_dict = SinkAdapter.dump_python(sink, exclude=SinkMapper._COMMON_FIELDS, exclude_none=True)

        return SinkDB(
            id=str(sink.id),
            name=sink.name,
            sink_type=sink.sink_type,
            output_formats=sink.output_formats,
            rate_limit=sink.rate_limit,
            config_data=sink_dict,
        )
