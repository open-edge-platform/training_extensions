// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Label } from '../../../../constants/shared-types';

export const toggleLabel = (newLabel: Label, labels: Label[]): Label[] => {
    const isExistingLabel = labels.some(({ id }) => id === newLabel.id);

    if (isExistingLabel) {
        return labels.filter(({ id }) => id !== newLabel.id) as Label[];
    }

    return [...labels, newLabel];
};

export const getNextItem = (totalItems: number, newIndex: number) => {
    return Math.min(totalItems, newIndex + 1);
};
