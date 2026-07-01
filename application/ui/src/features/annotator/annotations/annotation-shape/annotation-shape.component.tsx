// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useLabelResolver } from '../../../../shared/annotator/labels';
import type { Annotation } from '../../../../shared/types';
import { useSelectedMediaItem } from '../../selected-media-item-provider.component';
import { getFormattedPoints, isPrediction } from '../utils';

type FullImageShapeProps = {
    color: string;
    ariaLabel: string;
    strokeDasharray: string | undefined;
};

const FullImageShape = ({ color, ariaLabel, strokeDasharray }: FullImageShapeProps) => {
    const { mediaItem } = useSelectedMediaItem();

    return (
        <rect
            fill={'none'}
            stroke={color}
            aria-label={ariaLabel}
            width={mediaItem?.width}
            height={mediaItem?.height}
            strokeDasharray={strokeDasharray}
        />
    );
};

type AnnotationShapeProps = {
    annotation: Annotation;
};

export const AnnotationShape = ({ annotation }: AnnotationShapeProps) => {
    const { resolveAnnotationLabel } = useLabelResolver();
    const { shape, labels } = annotation;
    const hasMultipleLabels = labels.length > 1;
    const firstResolved = labels.length ? resolveAnnotationLabel(labels[0]) : undefined;
    const color = hasMultipleLabels ? 'white' : (firstResolved?.color ?? '--annotation-fill');
    const hasPredictionLabel = labels.some(isPrediction);
    const strokeDasharray = hasPredictionLabel ? 'calc(10 / var(--zoom-scale)) calc(6 / var(--zoom-scale))' : undefined;

    if (shape.type === 'full_image') {
        return (
            <FullImageShape
                color={color}
                strokeDasharray={strokeDasharray}
                ariaLabel={`${hasPredictionLabel ? 'prediction' : 'annotation'} full image`}
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
