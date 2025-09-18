// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '../api/client';
import { useProjectIdentifier } from './use-project-identifier.hook';

export const useProjectTask = () => {
    const projectId = useProjectIdentifier();
    const projectTask = $api.useQuery('get', '/api/projects/{project_id}', {
        params: { path: { project_id: projectId } },
    });

    return projectTask.data?.task.task_type || 'detection';
};
