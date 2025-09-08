// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useAnnotator } from '../annotator-provider.component';
import { Annotation, Point } from '../types';

type AnnotationShapeProps = {
    annotation: Annotation;
};

const getFormattedPoints = (points: Point[]): string => {
    return points.map(({ x, y }) => `${x},${y}`).join(' ');
};

export const AnnotationShape = ({ annotation }: AnnotationShapeProps) => {
    const shape = annotation.shape;
    const color = annotation.labels[0].color;
    const { setSelectedAnnotation, selectedAnnotation } = useAnnotator();

    const isSelected = selectedAnnotation?.id === annotation.id;

    const selectedStyles = {
        strokeWidth: `calc(2 / var(--zoom-scale))`,
        cursor: 'move',
        stroke: 'var(--energy-blue)',
    };

    if (shape.shapeType === 'rect') {
        return (
            <rect
                onClick={() => setSelectedAnnotation(annotation)}
                aria-label='annotation rect'
                x={shape.x}
                y={shape.y}
                width={shape.width}
                height={shape.height}
                fill={color}
                style={isSelected ? selectedStyles : undefined}
            />
        );
    }

    return (
        <polygon
            onClick={() => setSelectedAnnotation(annotation)}
            aria-label='annotation polygon'
            points={getFormattedPoints(shape.points)}
            fill={color}
        />
    );
};
