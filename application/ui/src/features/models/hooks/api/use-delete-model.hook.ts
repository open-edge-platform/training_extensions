// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQueryClient } from '@tanstack/react-query';

import { $api } from '../../../../api/client';
import { getQueryKey } from '../../../../query-client/query-client';

export const useDeleteModel = () => {
    const queryClient = useQueryClient();

    return $api.useMutation('delete', '/api/projects/{project_id}/models/{model_id}', {
        onSuccess: (
            _,
            {
                params: {
                    path: { project_id },
                },
            }
        ) => {
            return queryClient.invalidateQueries({
                queryKey: getQueryKey([
                    'get',
                    '/api/projects/{project_id}/models',
                    { params: { path: { project_id } } },
                ]),
            });
        },
    });
};
