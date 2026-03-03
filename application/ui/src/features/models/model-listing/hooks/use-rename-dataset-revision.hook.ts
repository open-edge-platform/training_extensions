// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQueryClient } from '@tanstack/react-query';

import { $api } from '../../../../api/client';
import { getQueryKey } from '../../../../query-client/query-client';

export const useRenameDatasetRevision = () => {
    const queryClient = useQueryClient();

    return $api.useMutation('patch', '/api/projects/{project_id}/dataset_revisions/{dataset_revision_id}', {
        onSuccess: (
            _,
            {
                params: {
                    path: { project_id, dataset_revision_id },
                },
            }
        ) => {
            return Promise.all([
                queryClient.invalidateQueries({
                    queryKey: getQueryKey([
                        'get',
                        '/api/projects/{project_id}/dataset_revisions/{dataset_revision_id}',
                        { params: { path: { project_id, dataset_revision_id } } },
                    ]),
                }),
                queryClient.invalidateQueries({
                    queryKey: getQueryKey([
                        'get',
                        '/api/projects/{project_id}/dataset_revisions',
                        { params: { path: { project_id } } },
                    ]),
                }),
            ]);
        },
    });
};
