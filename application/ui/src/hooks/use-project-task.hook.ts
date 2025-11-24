// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProject } from './api/project.hook';

export const useProjectTask = () => {
    const projectTask = useProject();

    return projectTask.data?.task.task_type || 'detection';
};
