// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Label } from '../../../../constants/shared-types';

export const toggleLabel = (newLabel: Label, labels: Label[]): Label[] => {
    const hasNewLabel = labels.some(({ id }) => id === newLabel.id);

    if (hasNewLabel) {
        return labels.filter(({ id }) => id !== newLabel.id) as Label[];
    }

    return [...labels, newLabel];
};
