// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { TaskType } from '../constants/shared-types';

export const getFormatOptions = (taskType: TaskType) => {
    const options: Record<TaskType, { label: string; value: string }[]> = {
        classification: [
            { label: 'Geti', value: 'geti' },
            { label: 'VOC', value: 'voc' },
        ],
        instance_segmentation: [
            { label: 'Geti', value: 'geti' },
            { label: 'COCO', value: 'coco' },
        ],
        detection: [
            { label: 'Geti', value: 'geti' },
            { label: 'YOLO', value: 'yolo' },
            { label: 'COCO', value: 'coco' },
        ],
    };

    return options[taskType];
};
