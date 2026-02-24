// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { components } from '../../api/openapi-spec';

type TaskType = components['schemas']['TaskType'];

export const getFormatOptions = (taskType: TaskType) => {
    const options: Record<TaskType, { label: string; value: string }[]> = {
        classification: [{ label: 'GETI', value: 'geti' }],
        instance_segmentation: [
            { label: 'GETI', value: 'geti' },
            { label: 'YOLO', value: 'yolo' },
        ],
        detection: [
            { label: 'GETI', value: 'geti' },
            { label: 'YOLO', value: 'yolo' },
            { label: 'COCO', value: 'coco' },
        ],
    };

    return options[taskType];
};
