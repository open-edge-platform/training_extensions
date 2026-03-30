// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { usePrefetchQuery, useQuery } from '@tanstack/react-query';

import { $api } from '../api/client';
import { useProjectIdentifier } from './use-project-identifier.hook';

const getDatasetRevisionsQueryOptions = (projectId: string) => {
    return $api.queryOptions('get', '/api/projects/{project_id}/dataset_revisions', {
        params: {
            path: {
                project_id: projectId,
            },
        },
    });
};

export const useGetDatasetRevisions = () => {
    const projectId = useProjectIdentifier();

    return useQuery(getDatasetRevisionsQueryOptions(projectId));
};

export const usePrefetchDatasetRevisions = () => {
    const projectId = useProjectIdentifier();

    return usePrefetchQuery(getDatasetRevisionsQueryOptions(projectId));
};
