// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { TaskType } from './task-selection/interface';

export const isClassificationTask = (taskType: TaskType | null): boolean => {
    return taskType === 'classification';
};
