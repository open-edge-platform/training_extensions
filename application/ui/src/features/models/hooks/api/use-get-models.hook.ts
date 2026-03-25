// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { usePrefetchQuery, useSuspenseQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../api/client';

const getModelsQueryOptions = (projectId: string) => {
    return $api.queryOptions('get', '/api/projects/{project_id}/models', {
        params: { path: { project_id: projectId } },
    });
};

export const useGetModels = () => {
    const projectId = useProjectIdentifier();

    return useSuspenseQuery(getModelsQueryOptions(projectId));
};

export const usePrefetchModels = () => {
    const projectId = useProjectIdentifier();

    return usePrefetchQuery(getModelsQueryOptions(projectId));
};
