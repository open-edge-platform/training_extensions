// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { SchemaModelView } from '../src/api/openapi-spec';

export const getMockedModel = (overrides: Partial<SchemaModelView> = {}): SchemaModelView => ({
    id: 'model-1',
    name: 'Mocked Model',
    architecture: 'YOLOX',
    parent_revision: null,
    training_info: {
        status: 'successful',
        label_schema_revision: {},
        configuration: {},
    },
    files_deleted: false,
    ...overrides,
});
