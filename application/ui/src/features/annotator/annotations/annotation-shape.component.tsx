// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Annotation, Point } from '../types';

type AnnotationShapeProps = {
    annotation: Annotation;
};

const getFormattedPoints = (points: Point[]): string => {
    return points.map(({ x, y }) => `${x},${y}`).join(' ');
};

export const AnnotationShape = ({ annotation }: AnnotationShapeProps) => {
    const { shape, labels } = annotation;
    const color = labels[0].color;

    if (shape.shapeType === 'rect') {
        return (
            <rect
                aria-label='annotation rect'
                x={shape.x}
                y={shape.y}
                width={shape.width}
                height={shape.height}
                fill={color}
            />
        );
    }

    return <polygon aria-label='annotation polygon' points={getFormattedPoints(shape.points)} fill={color} />;
};
