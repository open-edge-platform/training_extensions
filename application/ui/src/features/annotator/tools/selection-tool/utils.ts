// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Annotation, Point, Polygon, Rect, Shape } from '../../../../shared/types';
import { isNonEmptyArray } from '../../../../shared/util';

export const getTheTopShapeAt = (annotations: ReadonlyArray<Annotation>, point: Point): Annotation | null => {
    const intersectedAnnotations = annotations.filter((annotation: Annotation) => {
        const { shape } = annotation;

        return isPointInShape(shape, point);
    });

    if (isNonEmptyArray(intersectedAnnotations)) {
        return intersectedAnnotations[0];
    }

    return null;
};

export const isPointInShape = (shape: Shape, point: Point): boolean => {
    switch (shape.type) {
        case 'polygon':
            return pointInPolygon(shape, point);
        case 'rectangle':
            return pointInRectangle(shape, point);
        default:
            return false;
    }
};

export const pointInRectangle = ({ width, height, x, y }: Omit<Rect, 'shapeType'>, point: Point): boolean => {
    const startPoint: Point = { x, y };
    const endPoint: Point = { x: x + width, y: y + height };

    return point.x >= startPoint.x && point.x <= endPoint.x && point.y >= startPoint.y && point.y <= endPoint.y;
};

export const pointInPolygon = (polygon: Polygon, point: Point): boolean => {
    const polygonPoints = polygon.points;
    const pointsLength: number = polygonPoints.length;
    const x = point.x;
    const y = point.y;
    let inside = false;
    for (let i = 0, j = pointsLength - 1; i < pointsLength; j = i++) {
        const xi = polygonPoints[i].x;
        const yi = polygonPoints[i].y;
        const xj = polygonPoints[j].x;
        const yj = polygonPoints[j].y;

        const yDiffEquality = yi > y !== yj > y;
        const xDiff = xj - xi;
        const yiDiff = y - yi;
        const yijDiff = yj - yi;
        const intersect = yDiffEquality && x < (xDiff * yiDiff) / yijDiff + xi;

        if (intersect) inside = !inside;
    }
    return inside;
};

export const getIntersectedAnnotationsIds = (annotations: Annotation[], rect: Rect): string[] => {
    return annotations
        .filter((annotation: Annotation) => {
            const { shape } = annotation;

            if (shape.type === 'rectangle') {
                return rectanglesIntersect(shape, rect);
            } else if (shape.type === 'polygon') {
                return polygonIntersectsRectangle(shape, rect);
            }

            return false;
        })
        .map((annotation: Annotation) => annotation.id);
};

const rectanglesIntersect = (rect1: Rect, rect2: Rect): boolean => {
    return (
        rect1.x < rect2.x + rect2.width &&
        rect1.x + rect1.width > rect2.x &&
        rect1.y < rect2.y + rect2.height &&
        rect1.y + rect1.height > rect2.y
    );
};

const polygonIntersectsRectangle = (polygon: Polygon, rect: Rect): boolean => {
    const rectCorners: Point[] = [
        { x: rect.x, y: rect.y },
        { x: rect.x + rect.width, y: rect.y },
        { x: rect.x + rect.width, y: rect.y + rect.height },
        { x: rect.x, y: rect.y + rect.height },
    ];

    for (const point of polygon.points) {
        if (pointInRectangle(rect, point)) {
            return true;
        }
    }

    for (const corner of rectCorners) {
        if (pointInPolygon(polygon, corner)) {
            return true;
        }
    }

    return false;
};
