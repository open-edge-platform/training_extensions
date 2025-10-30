// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { $api } from 'src/api/client';

export const useProjects = () => {
    return $api.useSuspenseQuery('get', '/api/projects');
};

export const useProjectsQuery = () => {
    return $api.useQuery('get', '/api/projects');
};

export const useProject = () => {
    const projectId = useProjectIdentifier();

    return $api.useSuspenseQuery('get', '/api/projects/{project_id}', {
        params: { path: { project_id: projectId } },
    });
};

export const useCreateProject = () => {
    return $api.useMutation('post', '/api/projects', {
        meta: { invalidateQueries: [['get', '/api/projects']] },
    });
};

export const usePatchProject = () => {
    return $api.useMutation('patch', '/api/projects/{project_id}', {
        meta: { invalidateQueries: [['get', '/api/projects']] },
    });
};

export const useDeleteProject = () => {
    return $api.useMutation('delete', '/api/projects/{project_id}', {
        meta: { invalidateQueries: [['get', '/api/projects']] },
    });
};
