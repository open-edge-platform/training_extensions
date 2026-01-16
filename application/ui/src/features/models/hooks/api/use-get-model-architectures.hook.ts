// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../api/client';

export const useGetModelArchitectures = () => {
    return $api.useSuspenseQuery('get', '/api/model_architectures');
};

const useCurrentProject = () => {
    const projectId = useProjectIdentifier();

    return $api.useSuspenseQuery('get', '/api/projects/{project_id}', {
        params: { path: { project_id: projectId } },
    });
};

export const useGetTaskModelArchitectures = () => {
    const { data: projectData } = useCurrentProject();

    return $api.useSuspenseQuery('get', '/api/model_architectures', {
        params: {
            query: {
                task: projectData.task.task_type,
            },
        },
    });
};
