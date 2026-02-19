// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { type DatasetItem } from '../src/constants/shared-types';

export const getMockedDatasetItem = (overrides: Partial<DatasetItem>): DatasetItem => ({
    id: '1',
    subset: 'unassigned',
    user_reviewed: false,
    ...overrides,
});
