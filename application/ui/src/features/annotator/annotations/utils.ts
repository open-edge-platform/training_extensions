// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Annotation, AnnotationLabel, Point } from '../../../shared/types';

export const getFormattedPoints = (points: Point[]): string => {
    return points.map(({ x, y }) => `${x},${y}`).join(' ');
};

export const isRectangle = (annotation: Annotation): annotation is Annotation & { shape: { type: 'rectangle' } } => {
    return annotation.shape.type === 'rectangle';
};

export const isPolygon = (annotation: Annotation): annotation is Annotation & { shape: { type: 'polygon' } } => {
    return annotation.shape.type === 'polygon';
};

export const isPrediction = (
    label: Pick<AnnotationLabel, 'probability'>
): label is Required<Pick<AnnotationLabel, 'probability'>> => {
    return label.probability !== undefined;
};

export const convertPredictionToAnnotation = (prediction: Annotation): Annotation => {
    const convertedLabels = prediction.labels.map((label) => {
        if (isPrediction(label)) {
            const { probability, ...rest } = label;
            return rest;
        }
        return label;
    });

    return {
        ...prediction,
        labels: convertedLabels,
    };
};
