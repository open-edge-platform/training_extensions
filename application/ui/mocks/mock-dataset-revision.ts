// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { DatasetRevision } from '../src/constants/shared-types';

export const getMockedDatasetRevision = (overrides: Partial<DatasetRevision> = {}): DatasetRevision => ({
    id: 'dataset-1',
    created_at: '2025-01-01T00:00:00.000000+00:00',
    name: 'Dataset Revision 1',
    item_counts: {
        training: 70,
        validation: 20,
        testing: 10,
        total: 100,
    },
    files_deleted: false,
    size: 1048576,
    ...overrides,
});
