// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isObject } from 'lodash-es';

import { TaskType } from '../constants/shared-types';

export const getFormatOptions = (taskType: TaskType) => {
    const options: Record<TaskType, { label: string; value: string }[]> = {
        classification: [
            { label: 'GETI', value: 'geti' },
            { label: 'VOC', value: 'voc' },
        ],
        instance_segmentation: [
            { label: 'GETI', value: 'geti' },
            { label: 'COCO', value: 'coco' },
        ],
        detection: [
            { label: 'GETI', value: 'geti' },
            { label: 'YOLO', value: 'yolo' },
            { label: 'COCO', value: 'coco' },
        ],
    };

    return options[taskType];
};

export const getErrorMessage = (error: unknown) => {
    if (!isObject(error)) {
        return 'Unknown error';
    }

    if ('message' in error) {
        return String(error.message);
    }

    if ('detail' in error) {
        return String(error.detail);
    }

    return 'Unknown error';
};
