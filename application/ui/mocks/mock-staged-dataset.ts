// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { components, SchemaStagedDatasetView } from '../src/api/openapi-spec';

export const getMockedStagedDataset = (overrides: Partial<SchemaStagedDatasetView>) => ({
    id: 'staged-dataset-456',
    format: 'geti' as components['schemas']['DatasetFormat'],
    compressed: true,
    ready_for_export: true,
    ready_for_import: false,
    size: 17702642,
    metadata: null,
    ...overrides,
});
