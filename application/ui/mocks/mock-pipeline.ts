// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SchemaPipelineView } from './../src/api/openapi-spec.d';

export const getMockedPipeline = (customPipeline?: Partial<SchemaPipelineView>): SchemaPipelineView => {
    return {
        project_id: '123',
        status: 'running' as const,
        data_collection_policies: [],
        source: {
            id: 'source-id',
            name: 'source',
            source_type: 'video_file' as const,
            video_path: 'video.mp4',
        },
        model: {
            id: '1',
            name: 'My amazing model',
            architecture: 'Object_Detection_TestModel',
            training_info: {
                status: 'successful' as const,
                label_schema_revision: {},
                configuration: {},
            },
            files_deleted: false,
        },
        sink: {
            id: 'sink-id',
            name: 'sink',
            folder_path: 'data/sink',
            output_formats: ['image_original', 'image_with_predictions', 'predictions'] as Array<
                'image_original' | 'image_with_predictions' | 'predictions'
            >,
            rate_limit: 0.2,
            sink_type: 'folder' as const,
        },
        device: 'cpu',
        ...customPipeline,
    };
};
