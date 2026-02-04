// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQueryClient } from '@tanstack/react-query';

import { $api } from '../../../../api/client';
import { getQueryKey } from '../../../../query-client/query-client';

export const useUpdateLabel = () => {
    const queryClient = useQueryClient();

    return $api.useMutation('patch', '/api/projects/{project_id}/labels', {
        onSuccess: (
            _,
            {
                params: {
                    path: { project_id },
                },
            }
        ) => {
            return queryClient.invalidateQueries({
                queryKey: getQueryKey(['get', '/api/projects/{project_id}', { params: { path: { project_id } } }]),
            });
        },
    });
};
