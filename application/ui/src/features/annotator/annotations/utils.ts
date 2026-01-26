// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Annotation, Point } from '../../../shared/types';

export const getFormattedPoints = (points: Point[]): string => {
    return points.map(({ x, y }) => `${x},${y}`).join(' ');
};

export const isRectangle = (annotation: Annotation): annotation is Annotation & { shape: { type: 'rectangle' } } => {
    return annotation.shape.type === 'rectangle';
};

export const isPolygon = (annotation: Annotation): annotation is Annotation & { shape: { type: 'polygon' } } => {
    return annotation.shape.type === 'polygon';
};
