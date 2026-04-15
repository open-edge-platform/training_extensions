# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import pytest

from app.datumaro_converter.utils import ShapeConverter
from app.models import Point, Polygon, Rectangle


def test_rectangle_to_bbox() -> None:
    rectangle = Rectangle(x=10, y=20, width=200, height=100)
    result = ShapeConverter.rectangle_to_bbox(rectangle)
    assert result == [10, 20, 210, 120]


def test_polygon_to_points() -> None:
    polygon = Polygon(
        points=[Point(x=10.1, y=20.2), Point(x=20.2, y=30.3), Point(x=30.3, y=40.4), Point(x=40.4, y=50.5)]
    )
    result = ShapeConverter.polygon_to_points(polygon)
    expected = [[10.1, 20.2], [20.2, 30.3], [30.3, 40.4], [40.4, 50.5]]
    assert len(result) == len(expected)
    for actual_point, expected_point in zip(result, expected):
        assert actual_point == pytest.approx(expected_point, abs=1e-6)
