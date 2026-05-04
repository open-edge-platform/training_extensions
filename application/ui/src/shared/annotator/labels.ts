// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import { useProject } from 'hooks/api/project.hook';

import type { Label, TaskType } from '../../constants/shared-types';
import { isClassificationTask } from '../../features/project/task-type-guards';

export const EMPTY_LABEL_ID = 'empty-label';
const NO_LABEL: Label = { id: EMPTY_LABEL_ID, name: 'No label', color: '#FFF' };
const NO_OBJECT_LABEL: Label = { id: EMPTY_LABEL_ID, name: 'No object', color: '#FFF' };

export const isEmptyLabel = (id: string): boolean => id === EMPTY_LABEL_ID;

const getEmptyLabel = (taskType: TaskType, exclusiveLabels: boolean): Label | null => {
    if (isClassificationTask(taskType)) {
        const isMultiLabel = exclusiveLabels === false;

        if (isMultiLabel) {
            return NO_LABEL;
        }

        return null;
    }

    return NO_OBJECT_LABEL;
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

export const filterOutEmptyLabels = <T extends Pick<Label, 'id'>>(labels: T[]): T[] =>
    labels.filter((label) => label.id !== EMPTY_LABEL_ID);
