// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Annotation, Shape } from '../src/shared/types';
import { getMockedLabel } from './mock-labels';

const SHAPE_DEFAULTS = {
    rectangle: { type: 'rectangle', x: 10, y: 20, width: 100, height: 50 },
    polygon: { type: 'polygon', points: [] },
    full_image: { type: 'full_image' },
};

export const getMockedShape = (shape: Partial<Shape> & Pick<Shape, 'type'> = { type: 'rectangle' }): Shape => {
    return { ...SHAPE_DEFAULTS[shape.type], ...shape } as Shape;
};

export const getMockedAnnotation = (annotation?: Partial<Annotation>): Annotation => {
    return {
        id: 'annotation-1',
        shape: getMockedShape(),
        labels: [getMockedLabel()],
        ...annotation,
    };
};
