// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ExtendedModel, Model } from '../src/constants/shared-types';

export const getMockedModel = (overrides: Partial<Model> = {}): Model => ({
    id: '76e07d18-196e-4e33-bf98-ac1d35dca4cb',
    name: 'Object_Detection_YOLOX_X (76e07d18)',
    architecture: 'Object_Detection_YOLOX_X',
    parent_revision: null,
    size: 1048576,
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
    variants: [],
    files_deleted: false,
    ...overrides,
});

export const getMockedExtendedModel = (overrides: Partial<ExtendedModel> = {}): ExtendedModel => ({
    ...getMockedModel(overrides),
    evaluations: [
        {
            dataset_revision_id: '3c6c6d38-1cd8-4458-b759-b9880c048b78',
            subset: 'testing',
            metrics: [
                { name: 'accuracy', value: 0.97 },
                { name: 'precision', value: 0.98 },
                { name: 'recall', value: 0.94 },
            ],
        },
    ],
    ...overrides,
});
