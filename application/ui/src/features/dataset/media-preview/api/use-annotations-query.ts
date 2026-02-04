// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useSuspenseQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isObject } from 'lodash-es';

import { fetchClient } from '../../../../api/client';
import { AnnotationDTO } from '../../../../constants/shared-types';
import { EMPTY_LABEL_ID } from '../../../../shared/annotator/labels';

const isUnannotatedError = (error: unknown): boolean => {
    return (
        isObject(error) && 'detail' in error && /Dataset item has not been annotated yet/i.test(String(error.detail))
    );
};

export const useAnnotationsQuery = (datasetItemId: string) => {
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

            // If there are no annotations, it means that the user annotated the whole image with the empty label.
            if (data?.annotations.length === 0) {
                const annotationDTO = {
                    shape: {
                        type: 'full_image',
                    },
                    labels: [{ id: EMPTY_LABEL_ID }],
                } satisfies AnnotationDTO;

                return {
                    annotations: [annotationDTO],
                    user_reviewed: true,
                };
            }

            return data;
        },
    });
};
