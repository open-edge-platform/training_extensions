// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SchemaProjectView } from './../src/api/openapi-spec.d';

export const getMockedProject = (customProject: Partial<SchemaProjectView>): SchemaProjectView => {
    return {
        id: '7b073838-99d3-42ff-9018-4e901eb047fc',
        name: 'animals',
        task: {
            exclusive_labels: true,
            labels: [
                {
                    color: '#FF5733',
                    hotkey: 'S',
                    id: 'a22d82ba-afa9-4d6e-bbc1-8c8e4002ec29',
                    name: 'Object',
                },
            ],
            task_type: 'classification',
        },
        active_pipeline: false,
        ...customProject,
    };
};
