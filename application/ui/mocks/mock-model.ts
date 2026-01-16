// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { SchemaModelView } from '../src/api/openapi-spec';

export const getMockedModel = (overrides: Partial<SchemaModelView> = {}): SchemaModelView => ({
    id: '76e07d18-196e-4e33-bf98-ac1d35dca4cb',
    name: 'Object_Detection_YOLOX_X (76e07d18)',
    architecture: 'Object_Detection_YOLOX_X',
    parent_revision: null,
    training_info: {
        status: 'successful',
        label_schema_revision: {
            labels: [
                { id: 'a22d82ba-afa9-4d6e-bbc1-8c8e4002ec29', name: 'cat' },
                { id: '8aa85368-11ba-4507-88f2-6a6704d78ef5', name: 'dog' },
            ],
        },
        configuration: {},
        start_time: '2025-01-10T10:00:00.000000+00:00',
        end_time: '2025-01-10T12:30:00.000000+00:00',
        dataset_revision_id: '3c6c6d38-1cd8-4458-b759-b9880c048b78',
    },
    files_deleted: false,
    ...overrides,
});
