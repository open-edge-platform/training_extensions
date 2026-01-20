// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { TaskType } from '../../constants/shared-types';

export const isClassificationTask = (taskType: TaskType | null): boolean => {
    return taskType === 'classification';
};
