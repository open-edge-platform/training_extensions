// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useSuspenseQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isObject } from 'lodash-es';

import { fetchClient } from '../../../../api/client';

const isUnannotatedError = (error: unknown): boolean => {
    return (
        isObject(error) && 'detail' in error && /Dataset item has not been annotated yet/i.test(String(error.detail))
    );
};

export const useAnnotations = (datasetItemId: string) => {
    const projectId = useProjectIdentifier();

    return useSuspenseQuery({
        queryKey: [
            'get',
            `/api/projects/{project_id}/dataset/items/{dataset_item_id}/annotations`,
            { params: { path: { project_id: projectId, dataset_item_id: datasetItemId } } },
        ],
        queryFn: async () => {
            const { data, error } = await fetchClient.GET(
                '/api/projects/{project_id}/dataset/items/{dataset_item_id}/annotations',
                {
                    params: {
                        path: {
                            project_id: projectId,
                            dataset_item_id: datasetItemId,
                        },
                    },
                }
            );

            if (isUnannotatedError(error)) {
                return { annotations: [], user_reviewed: false };
            }

            if (error) {
                throw error;
            }

            return data;
        },
    });
};
