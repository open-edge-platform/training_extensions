// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Label } from '../../src/features/annotator/types';

export const getMockedLabel = (label?: Partial<Label>): Label => {
    return {
        color: 'red',
        id: 'label-1',
        name: 'label-1',
        isPrediction: false,
        ...label,
    };
};
