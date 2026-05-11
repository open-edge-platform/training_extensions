# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from sqlalchemy.engine import Dialect

from app.db.datetime import UTCDateTime


@pytest.fixture
def fxt_utc_dt() -> UTCDateTime:
    return UTCDateTime()


@pytest.fixture
def fxt_dialect() -> Mock:
    return Mock(spec=Dialect)


# --- process_bind_param ---


def test_bind_none_returns_none(fxt_utc_dt, fxt_dialect):
    assert fxt_utc_dt.process_bind_param(None, dialect=fxt_dialect) is None


def test_bind_naive_datetime_raises(fxt_utc_dt, fxt_dialect):
    with pytest.raises(ValueError, match="Naive datetime is not allowed"):
        fxt_utc_dt.process_bind_param(datetime(2026, 5, 11, 12, 0, 0), dialect=fxt_dialect)


def test_bind_utc_aware_strips_tzinfo(fxt_utc_dt, fxt_dialect):
    value = datetime(2026, 5, 11, 12, 0, 0, tzinfo=UTC)
    result = fxt_utc_dt.process_bind_param(value, dialect=fxt_dialect)
    assert result.tzinfo is None
    assert result == datetime(2026, 5, 11, 12, 0, 0)


def test_bind_non_utc_converts_to_utc(fxt_utc_dt, fxt_dialect):
    # +02:00 offset → subtract 2h to get UTC
    tz = timezone(timedelta(hours=2))
    value = datetime(2026, 5, 11, 14, 0, 0, tzinfo=tz)  # 14:00+02:00 = 12:00 UTC
    result = fxt_utc_dt.process_bind_param(value, dialect=fxt_dialect)
    assert result.tzinfo is None
    assert result == datetime(2026, 5, 11, 12, 0, 0)


# --- process_result_value ---


def test_result_none_returns_none(fxt_utc_dt, fxt_dialect):
    assert fxt_utc_dt.process_result_value(None, dialect=fxt_dialect) is None


def test_result_attaches_utc(fxt_utc_dt, fxt_dialect):
    naive = datetime(2026, 5, 11, 12, 0, 0)
    result = fxt_utc_dt.process_result_value(naive, dialect=fxt_dialect)
    assert result.tzinfo == UTC
    assert result == datetime(2026, 5, 11, 12, 0, 0, tzinfo=UTC)


# --- round-trip ---


def test_round_trip_preserves_value(fxt_utc_dt, fxt_dialect):
    original = datetime(2026, 5, 11, 12, 0, 0, tzinfo=UTC)
    stored = fxt_utc_dt.process_bind_param(original, dialect=fxt_dialect)
    recovered = fxt_utc_dt.process_result_value(stored, dialect=fxt_dialect)
    assert recovered == original
    assert recovered.tzinfo == UTC


def test_round_trip_non_utc_normalizes(fxt_utc_dt, fxt_dialect):
    tz = timezone(timedelta(hours=5, minutes=30))
    original = datetime(2026, 5, 11, 17, 30, 0, tzinfo=tz)  # 17:30+05:30 = 12:00 UTC
    stored = fxt_utc_dt.process_bind_param(original, dialect=fxt_dialect)
    recovered = fxt_utc_dt.process_result_value(stored, dialect=fxt_dialect)
    assert recovered == datetime(2026, 5, 11, 12, 0, 0, tzinfo=UTC)
