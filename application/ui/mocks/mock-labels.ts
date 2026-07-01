// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Label } from '../src/constants/shared-types';
import type { AnnotationLabel, AnnotationLabelRef } from '../src/shared/types';

export const getMockedLabel = (label?: Partial<Label>): Label & { isPrediction: boolean } => {
    return {
        color: '#ffff00',
        id: 'label-1',
        name: 'label-1',
        isPrediction: false,
        ...label,
    };
};

export const getMockedAnnotationLabel = (label?: Partial<AnnotationLabel>): AnnotationLabel => {
    return {
        color: '#ffff00',
        id: 'label-1',
        name: 'label-1',
        ...label,
    };
};

export const getMockedAnnotationLabelRef = (ref?: Partial<AnnotationLabelRef>): AnnotationLabelRef => {
    return {
        id: 'label-1',
        ...ref,
    };
};
