// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isObject } from 'lodash-es';

import { fetchClient } from '../../../../api/client';
import type { AnnotationDTO, Media } from '../../../../constants/shared-types';
import { getQueryKey } from '../../../../query-client/query-client';
import { EMPTY_LABEL_ID } from '../../../../shared/annotator/labels';
import { isVideoFrame } from '../../../../shared/media-item-utils';

const isUnannotatedError = (error: unknown): boolean => {
    return isObject(error) && 'detail' in error && /Media has not been annotated yet/i.test(String(error.detail));
};

export const useAnnotationsQuery = (media: Media) => {
    const projectId = useProjectIdentifier();

    const queryParams = isVideoFrame(media)
        ? {
              frame_index: media.frame_number,
          }
        : undefined;

    return useQuery({
        queryKey: getQueryKey([
            'get',
            `/api/projects/{project_id}/dataset/media/{media_id}/annotations`,
            {
                params: {
                    path: { project_id: projectId, media_id: media.id },
                    query: queryParams,
                },
            },
        ]),
        queryFn: async () => {
            const { data, error, response } = await fetchClient.GET(
                '/api/projects/{project_id}/dataset/media/{media_id}/annotations',
                {
                    params: {
                        path: {
                            project_id: projectId,
                            media_id: media.id,
                        },
                        query: queryParams,
                    },
                }
            );

            if (isUnannotatedError(error) || response.status === 404) {
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
