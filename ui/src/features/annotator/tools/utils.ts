// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Point, RegionOfInterest } from '../types';

type ElementType = SVGElement | HTMLDivElement;
export const getRelativePoint = (element: ElementType, point: Point, zoom: number): Point => {
    const rect = element.getBoundingClientRect();

    return {
        x: Math.round((point.x - rect.left) / zoom),
        y: Math.round((point.y - rect.top) / zoom),
    };
};

export const getBoundingBoxInRoi = (boundingBox: RegionOfInterest, roi: RegionOfInterest) => {
    const x = Math.max(0, boundingBox.x);
    const y = Math.max(0, boundingBox.y);

    return {
        x,
        y,
        width: Math.min(roi.width - x, boundingBox.width),
        height: Math.min(roi.height - y, boundingBox.height),
    };
};

interface getBoundingBoxResizePointsProps {
    gap: number;
    boundingBox: RegionOfInterest;
    onResized: (boundingBox: RegionOfInterest) => void;
}
export const getBoundingBoxResizePoints = ({ boundingBox, gap, onResized }: getBoundingBoxResizePointsProps) => {
    return [
        {
            x: boundingBox.x,
            y: boundingBox.y,
            moveAnchorTo: (x: number, y: number) => {
                const x1 = Math.max(0, Math.min(x, boundingBox.x + boundingBox.width - gap));
                const y1 = Math.max(0, Math.min(y, boundingBox.y + boundingBox.height - gap));

                onResized({
                    x: x1,
                    width: Math.max(gap, boundingBox.width + boundingBox.x - x1),
                    y: y1,
                    height: Math.max(gap, boundingBox.height + boundingBox.y - y1),
                });
            },
            cursor: 'nw-resize',
            label: 'North west resize anchor',
        },
        {
            x: boundingBox.x + boundingBox.width / 2,
            y: boundingBox.y,
            moveAnchorTo: (_x: number, y: number) => {
                const y1 = Math.max(0, Math.min(y, boundingBox.y + boundingBox.height - gap));

                onResized({
                    ...boundingBox,
                    y: y1,
                    height: Math.max(gap, boundingBox.height + boundingBox.y - y1),
                });
            },
            cursor: 'n-resize',
            label: 'North resize anchor',
        },
        {
            x: boundingBox.x + boundingBox.width,
            y: boundingBox.y,
            moveAnchorTo: (x: number, y: number) => {
                const y1 = Math.max(0, Math.min(y, boundingBox.y + boundingBox.height - gap));

                onResized({
                    ...boundingBox,
                    width: Math.max(gap, x - boundingBox.x),
                    y: y1,
                    height: Math.max(gap, boundingBox.height + boundingBox.y - y1),
                });
            },
            cursor: 'ne-resize',
            label: 'North east resize anchor',
        },
        {
            x: boundingBox.x + boundingBox.width,
            y: boundingBox.y + boundingBox.height / 2,
            moveAnchorTo: (x: number) => {
                onResized({ ...boundingBox, width: Math.max(gap, x - boundingBox.x) });
            },
            cursor: 'e-resize',
            label: 'East resize anchor',
        },
        {
            x: boundingBox.x + boundingBox.width,
            y: boundingBox.y + boundingBox.height,
            moveAnchorTo: (x: number, y: number) => {
                onResized({
                    x: boundingBox.x,
                    width: Math.max(gap, x - boundingBox.x),

                    y: boundingBox.y,
                    height: Math.max(gap, y - boundingBox.y),
                });
            },
            cursor: 'se-resize',
            label: 'South east resize anchor',
        },
        {
            x: boundingBox.x + boundingBox.width / 2,
            y: boundingBox.y + boundingBox.height,
            moveAnchorTo: (_x: number, y: number) => {
                onResized({
                    ...boundingBox,
                    y: boundingBox.y,
                    height: Math.max(gap, y - boundingBox.y),
                });
            },
            cursor: 's-resize',
            label: 'South resize anchor',
        },
        {
            x: boundingBox.x,
            y: boundingBox.y + boundingBox.height,
            moveAnchorTo: (x: number, y: number) => {
                const x1 = Math.max(0, Math.min(x, boundingBox.x + boundingBox.width - gap));

                onResized({
                    x: x1,
                    width: Math.max(gap, boundingBox.width + boundingBox.x - x1),

                    y: boundingBox.y,
                    height: Math.max(gap, y - boundingBox.y),
                });
            },
            cursor: 'sw-resize',
            label: 'South west resize anchor',
        },
        {
            x: boundingBox.x,
            y: boundingBox.y + boundingBox.height / 2,
            moveAnchorTo: (x: number, _y: number) => {
                const x1 = Math.max(0, Math.min(x, boundingBox.x + boundingBox.width - gap));

                onResized({
                    ...boundingBox,
                    x: x1,
                    width: Math.max(gap, boundingBox.width + boundingBox.x - x1),
                });
            },
            cursor: 'w-resize',
            label: 'West resize anchor',
        },
    ];
};

const clampBetween = (min: number, value: number, max: number): number => {
    return Math.max(min, Math.min(max, value));
};

export const getClampedBoundingBox = (point: Point, boundingBox: RegionOfInterest, roi: RegionOfInterest) => {
    const roiX = roi.width + roi.x;
    const roiY = roi.height + roi.y;
    const shapeX = boundingBox.width + boundingBox.x;
    const shapeY = boundingBox.height + boundingBox.y;

    const clampedTranslate = {
        x: clampBetween(shapeX - roiX, -point.x, boundingBox.x - roi.x),
        y: clampBetween(shapeY - roiY, -point.y, boundingBox.y - roi.y),
    };

    return {
        ...boundingBox,
        x: boundingBox.x - clampedTranslate.x,
        y: boundingBox.y - clampedTranslate.y,
    };
};
