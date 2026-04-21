// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { AnnotationType, TaskType } from '../../../../../constants/shared-types';

export const TASK_SELECTION_FORM_ID = 'task-selection-form';

export const getRecommendedTaskType = (annotationType: AnnotationType | undefined): TaskType | undefined => {
    switch (annotationType) {
        case 'bounding_box':
            return 'detection';
        case 'polygon':
            return 'instance_segmentation';
        case 'label':
            return 'classification';
        default:
            return undefined;
    }
};
