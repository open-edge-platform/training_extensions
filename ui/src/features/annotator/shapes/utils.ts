// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type ClipperShape from '@doodle3d/clipper-js';
import Clipper from '@doodle3d/clipper-js';
import { Shape as SmartToolsShape, Polygon as ToolPolygon, Rect as ToolRect } from '@geti/smart-tools/types';
import { BoundingBox } from '@geti/smart-tools/utils';
import { isEmpty } from 'lodash-es';

import { ClipperPoint, Point, Polygon, Rect, RegionOfInterest, Shape } from '../types';

// @ts-expect-error `default` actually exists in the module
const ClipperJS = Clipper.default || Clipper;

export function convertToolShapeToGetiShape(shape: ToolPolygon): Polygon;
export function convertToolShapeToGetiShape(shape: ToolRect): Rect;
export function convertToolShapeToGetiShape(shape: SmartToolsShape): Shape;
export function convertToolShapeToGetiShape(shape: SmartToolsShape): Shape {
    switch (shape.shapeType) {
        case 'polygon':
            return { shapeType: 'polygon', points: shape.points };
        case 'rect':
            return {
                shapeType: 'rect',
                x: shape.x,
                y: shape.y,
                width: shape.width,
                height: shape.height,
            };
        default:
            throw new Error('Unknown shape type');
    }
}

const removeOffPointsRect = (rect: Rect, roi: RegionOfInterest): Rect => {
    const { x, y, width, height } = roi;

    let newRect: Rect = {
        ...rect,
    };

    if (rect.x < x) {
        newRect = {
            ...newRect,
            x,
            width: newRect.width - (x - rect.x),
        };
    }

    if (rect.x + rect.width > x + width) {
        const diff = rect.x + rect.width - x - width;

        newRect = {
            ...newRect,
            width: rect.width - diff,
        };
    }

    if (rect.y < y) {
        newRect = {
            ...newRect,
            y,
            height: rect.height - (y - rect.y),
        };
    }

    if (newRect.y + rect.height > y + height) {
        const diff = rect.y + rect.height - y - height;

        newRect = {
            ...newRect,
            height: rect.height - diff,
        };
    }

    return newRect;
};

const removeOffLimitPointsPolygon = (shape: Shape, roi: RegionOfInterest): Polygon => {
    const { width, height, x, y } = roi;
    const getRect = (rx: number, ry: number, rWidth: number, rHeight: number): Rect => ({
        x: rx,
        y: ry,
        width: rWidth,
        height: rHeight,
        shapeType: 'rect',
    });
    // `eraserSize` Builds and positions rect shapes around ROI limits (top, left, right, bottom),
    // finally `getShapesDifference` will use those rects to calc and remove offline polygons
    const eraserSize = 10;
    const topRect = getRect(x - eraserSize, y - eraserSize, width + eraserSize * 3, eraserSize);
    const leftRect = getRect(x - eraserSize, y - eraserSize, eraserSize, height * 2);
    const rightRect = getRect(x + width, y - eraserSize, eraserSize, height * 2);
    const bottomRect = getRect(x - eraserSize, y + height, width + eraserSize * 3, eraserSize);

    return [leftRect, bottomRect, rightRect, topRect].reduce(
        (accum, current) => getShapesDifference(roi, accum, current),
        shape
    ) as Polygon;
};

const calculateRectanglePoints = (shape: BoundingBox): ClipperPoint[] => {
    const { x: X, y: Y, width, height } = shape;
    const topLeftPoint = { X, Y };
    const topRightPoint = { X: X + width, Y };
    const bottomLeftPoint = { X, Y: Y + height };
    const bottomRightPoint = { X: X + width, Y: Y + height };

    return [topLeftPoint, topRightPoint, bottomRightPoint, bottomLeftPoint];
};

const convertPolygonPoints = (shape: Polygon): ClipperPoint[] => {
    return shape.points.map(({ x, y }: Point) => ({ X: x, Y: y }));
};

const transformToClipperShape = (shape: Shape): ClipperShape => {
    switch (true) {
        case shape.shapeType === 'rect':
            return new ClipperJS([calculateRectanglePoints(shape)], true);
        default:
            return new ClipperJS([convertPolygonPoints(shape)], true);
    }
};

const runUnionOrDifference =
    <T>(algorithm: 'union' | 'difference', formatTo: (path: ClipperPoint[]) => T) =>
    (roi: RegionOfInterest, subj: Shape, clip: Shape): T => {
        const subjShape = transformToClipperShape(subj);
        const clipShape = transformToClipperShape(clip);
        const solutionPath = subjShape[algorithm](clipShape);
        const filteredPath = filterIntersectedPathsWithRoi(roi, solutionPath);
        const biggestPath = findBiggerSubPath(filteredPath);

        return formatTo(biggestPath);
    };

const clipperShapeToPolygon = (path: ClipperPoint[]): Polygon => ({
    shapeType: 'polygon',
    points: path.map(({ X, Y }) => ({ x: X, y: Y })),
});

export const getShapesDifference = runUnionOrDifference<Polygon>('difference', clipperShapeToPolygon);

const findBiggerSubPath = (shape: ClipperShape): ClipperPoint[] => {
    const areas = shape.areas();
    const { index: shapeIndex } = areas.reduce(
        (accum: { value: number; index: number }, value, index) => {
            return value > accum.value ? { value, index } : accum;
        },
        { value: 0, index: 0 }
    );

    return shape.paths.length ? shape.paths[shapeIndex] : [];
};

const hasIntersection = (clip: ClipperShape, subj: ClipperShape) => {
    const { paths } = clip.intersect(subj);

    return !isEmpty(paths);
};

const filterIntersectedPathsWithRoi = (roi: RegionOfInterest, shape: ClipperShape): ClipperShape => {
    const newPath = shape.clone();
    const roiRect = transformToClipperShape({ ...roi, shapeType: 'rect' });

    newPath.paths = newPath.paths.filter((subPath) => hasIntersection(roiRect, new ClipperJS([subPath])));

    return newPath;
};

export const removeOffLimitPoints = (shape: Shape, roi: RegionOfInterest): Shape => {
    return shape.shapeType === 'rect' ? removeOffPointsRect(shape, roi) : removeOffLimitPointsPolygon(shape, roi);
};

type ElementType = SVGElement | HTMLDivElement;
export const getRelativePoint = (element: ElementType, point: Point, zoom: number): Point => {
    const rect = element.getBoundingClientRect();

    return {
        x: Math.round((point.x - rect.left) / zoom),
        y: Math.round((point.y - rect.top) / zoom),
    };
};
