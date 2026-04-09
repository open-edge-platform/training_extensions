// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Annotation, Shape } from '../src/shared/types';
import { getMockedLabel } from './mock-labels';

export const getMockedShape = (shape: Partial<Shape> = {}): Shape => {
    return {
        ...shape,
        type: 'rectangle',
        x: 10,
        y: 20,
        width: 100,
        height: 50,
    };
};

export const getMockedAnnotation = (annotation?: Partial<Annotation>): Annotation => {
    return {
        id: 'annotation-1',
        shape: getMockedShape(),
        labels: [getMockedLabel()],
        ...annotation,
    };
};
