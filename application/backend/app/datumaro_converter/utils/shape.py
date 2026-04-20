# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from app.models import Polygon, Rectangle


class ShapeConverter:
    """Knows how to convert shapes to coordinate lists."""

    @staticmethod
    def rectangle_to_bbox(rectangle: Rectangle) -> list[int]:
        """Converts rectangle to x1y1x2y2 format."""
        return [rectangle.x, rectangle.y, rectangle.x + rectangle.width, rectangle.y + rectangle.height]

    @staticmethod
    def polygon_to_points(polygon: Polygon) -> list[list[float]]:
        """Converts polygon to list of xy points."""
        return [[point.x, point.y] for point in polygon.points]
