// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Label } from '../src/constants/shared-types';

export const getMockedLabel = (label?: Partial<Label>): Label & { isPrediction: boolean } => {
    return {
        color: '#ffff00',
        id: 'label-1',
        name: 'label-1',
        isPrediction: false,
        ...label,
    };
};
