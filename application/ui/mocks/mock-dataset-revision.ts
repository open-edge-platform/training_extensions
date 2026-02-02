// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { DatasetRevision } from '../src/constants/shared-types';

export const getMockedDatasetRevision = (overrides: Partial<DatasetRevision> = {}): DatasetRevision => ({
    id: 'dataset-1',
    project_id: 'project-1',
    name: 'Dataset Revision 1',
    item_counts: {
        training: 70,
        validation: 20,
        testing: 10,
        total: 100,
    },
    files_deleted: false,
    ...overrides,
});
