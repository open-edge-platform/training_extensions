// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Label } from '../../../constants/shared-types';

export const MAX_LABEL_NAME_LENGTH = 100;

const isUniqueLabelName = (name: string, existingLabels: Label[], excludeId?: string): boolean => {
    return !existingLabels.some((label) => label.name === name && label.id !== excludeId);
};

export const validateLabelName = (name: string, existingLabels: Label[], excludeId?: string): string | undefined => {
    const trimmedName = name.trim();

    if (!isUniqueLabelName(trimmedName, existingLabels, excludeId)) {
        return 'Label name must be unique.';
    }

    return undefined;
};
