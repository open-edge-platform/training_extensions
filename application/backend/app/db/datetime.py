# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, TypeDecorator
from sqlalchemy.engine import Dialect


class UTCDateTime(TypeDecorator):
    """
    SQLAlchemy custom type that enforces UTC-aware datetimes.
    https://docs.sqlalchemy.org/en/20/core/custom_types.html#store-timezone-aware-timestamps-as-timezone-naive-utc

    Stores datetime values in the database as naive UTC timestamps (compatible with SQLite's DATETIME type),
    while ensuring that all Python-side values are timezone-aware datetimes in UTC.

    Behaviour:
    - **Write (Python → DB):** Converts any tz-aware datetime to UTC and strips tzinfo before storing.
        Raises ``ValueError`` if a naive datetime is provided.
    - **Read (DB → Python):** Annotates the returned naive datetime with ``timezone.utc``, making it tz-aware.

    Raises:
        ValueError: If a naive datetime (without tzinfo) is passed on write.

    Example::

        class MyModel(Base):
            created_at: Mapped[datetime] = mapped_column(UTCDateTime())

        # Write — must be tz-aware
        item.created_at = datetime.now(UTC)          # OK
        item.created_at = datetime(2025, 1, 1)       # raises ValueError

        # Read — always returns UTC-aware datetime
        assert item.created_at.tzinfo == timezone.utc
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("Naive datetime is not allowed")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: Any, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC)
