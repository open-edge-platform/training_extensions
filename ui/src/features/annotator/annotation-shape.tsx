// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Annotation, Polygon as PolygonType } from './types';

const getFormattedPoints = (points: Array<{ x: number; y: number }>): string =>
    points.map(({ x, y }) => `${x},${y}`).join(' ');

const Polygon = ({ shape }: { shape: PolygonType }) => {
    const points = getFormattedPoints(shape.points);

    return <polygon points={points} />;
};

export const AnnotationShape = ({ annotation }: { annotation: Annotation }) => {
    const shape = annotation.shape;

    if (shape.type === 'bounding-box') {
        return (
            <rect x={shape.x} y={shape.y} width={shape.width} height={shape.height} fill={annotation.labels[0].color} />
        );
    }

    if (shape.type === 'polygon') {
        return <Polygon shape={shape} />;
    }

    if (shape.type === 'circle') {
        return <circle cx={shape.cx} cy={shape.cy} r={shape.r} />;
    }
};
