// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQueryClient } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../../api/client';
import { getQueryKey } from '../../../../../query-client/query-client';

export const useAssignLabel = () => {
    const projectId = useProjectIdentifier();
    const queryClient = useQueryClient();
    const mutation = $api.useMutation('post', '/api/projects/{project_id}/dataset/media/{media_id}/annotations');

    const invalidateQueries = () => {
        queryClient.invalidateQueries({
            queryKey: getQueryKey([
                'get',
                '/api/projects/{project_id}/dataset/items',
                { params: { path: { project_id: projectId } } },
            ]),
        });
        queryClient.invalidateQueries({
            queryKey: getQueryKey([
                'get',
                '/api/projects/{project_id}/dataset/media',
                {
                    params: {
                        path: { project_id: projectId },
                    },
                },
            ]),
        });
    };

    const assignLabel = async (mediaId: string, labelIds: string[]) => {
        return mutation.mutateAsync({
            params: {
                path: {
                    project_id: projectId,
                    media_id: mediaId,
                },
            },
            body: {
                annotations: [
                    {
                        shape: {
                            type: 'full_image',
                        },
                        labels: labelIds.map((id) => ({ id })),
                    },
                ],
            },
        });
    };

    return {
        mutate: assignLabel,
        isPending: mutation.isPending,
        invalidateQueries,
    };
};
