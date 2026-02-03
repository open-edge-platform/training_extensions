// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import { useProject } from 'hooks/api/project.hook';

import type { Label, TaskType } from '../../constants/shared-types';
import { isClassificationTask } from '../../features/project/task-type-guards';

export const EMPTY_LABEL_ID = 'empty-label';
export const NO_LABEl: Label = { id: EMPTY_LABEL_ID, name: 'No label', color: '#FFF' };
export const NO_OBJECTS_LABEL: Label = { id: EMPTY_LABEL_ID, name: 'No objects', color: '#FFF' };

const getEmptyLabel = (taskType: TaskType, exclusiveLabels: boolean): Label | null => {
    if (isClassificationTask(taskType)) {
        const isMultiLabel = exclusiveLabels === false;

        if (isMultiLabel) {
            return NO_LABEl;
        }

        return null;
    }

    return NO_OBJECTS_LABEL;
};

export const useProjectLabelsWithEmptyLabel = (): Label[] => {
    const { data: project } = useProject();
    const { labels = [], exclusive_labels, task_type } = project.task;

    return useMemo(() => {
        const label = getEmptyLabel(task_type, exclusive_labels);
        if (label) {
            return [...labels, label];
        }

        return labels;
    }, [exclusive_labels, labels, task_type]);
};
