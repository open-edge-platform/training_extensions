// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ModelVariant } from '../src/constants/shared-types';

export const getMockedVariant = (overrides: Partial<ModelVariant> = {}): ModelVariant => ({
    id: 'variant-id',
    format: 'openvino',
    precision: 'fp16',
    weights_size: 1024,
    evaluations: [],
    files_deleted: false,
    ...overrides,
});
