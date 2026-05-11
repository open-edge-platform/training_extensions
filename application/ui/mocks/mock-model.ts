// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Model, ModelArchitectureWithPerformanceCategory } from '../src/constants/shared-types';

export const getMockedModel = (overrides: Partial<Model> = {}): Model => {
    return {
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
            start_time: '2025-01-10T10:00:00.000000+00:00',
            end_time: '2025-01-10T12:30:00.000000+00:00',
            dataset_revision_id: '3c6c6d38-1cd8-4458-b759-b9880c048b78',
        },
        variants: [],
        files_deleted: false,
        ...overrides,
    };
};

export const getMockedModelArchitecture = (
    overrides: Partial<ModelArchitectureWithPerformanceCategory> = {}
): ModelArchitectureWithPerformanceCategory => ({
    id: 'Object_Detection_Deim_DFine_L',
    task: 'detection',
    name: 'Deim-DFine-L',
    license: "Apache 2.0",
    description: 'DEIM is an advanced training framework designed to enhance the matching mechanism in DETRs.',
    capabilities: {
        xai: true,
        tiling: true,
    },
    stats: {
        gigaflops: 91,
        trainable_parameters: 31,
        benchmark_metrics: {
            imagenet_top1_accuracy: 2,
            imagenet_top5_accuracy: 5,
            coco_map_50_95: 2,
            coco_map_50: 2,
        },
    },
    support_status: 'active',
    ...overrides,
});
