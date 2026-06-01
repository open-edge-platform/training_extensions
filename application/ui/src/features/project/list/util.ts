// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import dayjs from 'dayjs';

import { Task, TaskType } from '../../../constants/shared-types';
import { isMultiLabelClassificationTask } from '../task-type-guards';

export const formatCreationDate = (creationDate: string) => {
    return dayjs(creationDate).format('D MMMM YYYY | h:mm A');
};

export const MAP_PROJECT_TYPE_TO_TITLE: Record<TaskType, string> = {
    detection: 'Object detection',
    classification: 'Classification',
    instance_segmentation: 'Instance segmentation',
};

export const getProjectTypeTitle = (task?: Task): string | undefined => {
    if (task === undefined) {
        return undefined;
    }

    return isMultiLabelClassificationTask(task)
        ? 'Multi-label classification'
        : MAP_PROJECT_TYPE_TO_TITLE[task.task_type];
};
