// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQueryClient } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isEmpty } from 'lodash-es';

import { $api } from '../../../../../api/client';
import { getQueryKey } from '../../../../../query-client/query-client';
import { EMPTY_LABEL_ID } from '../../../../../shared/annotator/labels';

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
        const labelsWithoutEmptyLabel = labelIds.filter((id) => id !== EMPTY_LABEL_ID);

        return mutation.mutateAsync({
            params: {
                path: {
                    project_id: projectId,
                    media_id: mediaId,
                },
            },
            body: {
                annotations: isEmpty(labelsWithoutEmptyLabel)
                    ? []
                    : [{ shape: { type: 'full_image' }, labels: labelsWithoutEmptyLabel.map((id) => ({ id })) }],
            },
        });
    };

    return {
        mutate: assignLabel,
        isPending: mutation.isPending,
        invalidateQueries,
    };
};
