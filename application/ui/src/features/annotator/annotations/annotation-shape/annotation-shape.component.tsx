// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Annotation, Polygon } from '../../../../shared/types';
import { getFormattedPoints, isPrediction } from '../utils';

type AnnotationShapeProps = {
    annotation: Annotation;
};

export const AnnotationShape = ({ annotation }: AnnotationShapeProps) => {
    const { shape, labels } = annotation;
    const color = labels.length ? labels[0].color : '--annotation-fill';
    const strokeDasharray = annotation.labels.some(isPrediction)
        ? 'calc(10 / var(--zoom-scale)) calc(6 / var(--zoom-scale))'
        : undefined;

    if (shape.type === 'rectangle') {
        return (
            <rect
                aria-label='annotation rect'
                x={shape.x}
                y={shape.y}
                width={shape.width}
                height={shape.height}
                fill={color}
                strokeDasharray={strokeDasharray}
            />
        );
    }

    return (
        <polygon
            aria-label='annotation polygon'
            points={getFormattedPoints((shape as Polygon).points)}
            fill={color}
            strokeDasharray={strokeDasharray}
        />
    );
};
