// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Label } from '../../../constants/shared-types';

export const validateLabel = (label: Label, labels: Label[]): string | undefined => {
    if (label.name.trim().length === 0) {
        return 'Label name cannot be empty';
    }

    if (labels.some((l) => l.name === label.name)) {
        return 'That label name already exists';
    }

    return undefined;
};
