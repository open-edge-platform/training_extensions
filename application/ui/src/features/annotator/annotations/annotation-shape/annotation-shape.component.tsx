// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Annotation } from '../../../../shared/types';
import { useSelectedData } from '../../../dataset/selected-data-provider.component';
import { getFormattedPoints, isPrediction } from '../utils';

type AnnotationShapeProps = {
    annotation: Annotation;
};

export const AnnotationShape = ({ annotation }: AnnotationShapeProps) => {
    const { shape, labels } = annotation;
    const { selectedMediaItem } = useSelectedData();
    const color = labels.length ? labels[0].color : '--annotation-fill';
    const hasPredictionLabel = labels.some(isPrediction);
    const strokeDasharray = hasPredictionLabel ? 'calc(10 / var(--zoom-scale)) calc(6 / var(--zoom-scale))' : undefined;

    if (shape.type === 'full_image') {
        return (
            <rect
                fill={'none'}
                stroke={color}
                aria-label={`${hasPredictionLabel ? 'prediction' : 'annotation'} full image`}
                width={selectedMediaItem?.width}
                height={selectedMediaItem?.height}
                strokeDasharray={strokeDasharray}
            />
        );
    }

    if (shape.type === 'rectangle') {
        return (
            <rect
                aria-label={`${hasPredictionLabel ? 'prediction' : 'annotation'} rect`}
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
            aria-label={`${hasPredictionLabel ? 'prediction' : 'annotation'} polygon`}
            points={getFormattedPoints(shape.points)}
            fill={color}
            strokeDasharray={strokeDasharray}
        />
    );
};
