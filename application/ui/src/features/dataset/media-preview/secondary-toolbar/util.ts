// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Label } from '../../../../constants/shared-types';
import type { AnnotationLabelRef } from '../../../../shared/types';

export const toggleLabel = (newLabel: Label, labels: AnnotationLabelRef[]): AnnotationLabelRef[] => {
    const isExistingLabel = labels.some(({ id }) => id === newLabel.id);

    if (isExistingLabel) {
        return labels.filter(({ id }) => id !== newLabel.id);
    }

    return [...labels, { id: newLabel.id }];
};

export const getNextItem = (totalItems: number, newIndex: number) => {
    return Math.min(totalItems, newIndex + 1);
};
