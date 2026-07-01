// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { queryOptions, useQuery, type QueryKey } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isObject } from 'lodash-es';

import { fetchClient } from '../../../../api/client';
import type { AnnotationDTO, DatasetSubset, Media } from '../../../../constants/shared-types';
import { getQueryKey } from '../../../../query-client/query-client';
import { EMPTY_LABEL_ID } from '../../../../shared/annotator/labels';
import { isVideoFrame } from '../../../../shared/media-item-utils';

const isUnannotatedError = (error: unknown): boolean => {
    return isObject(error) && 'detail' in error && /Media has not been annotated yet/i.test(String(error.detail));
};

const getAnnotationsQueryParams = (media: Media) => {
    return isVideoFrame(media)
        ? {
              frame_index: media.frame_number,
          }
        : undefined;
};

const getAnnotationsQueryKey = (projectId: string, media: Media): QueryKey => {
    return getQueryKey([
        'get',
        `/api/projects/{project_id}/dataset/media/{media_id}/annotations`,
        {
            params: {
                path: { project_id: projectId, media_id: media.id },
                query: getAnnotationsQueryParams(media),
            },
        },
    ]);
};

export const annotationsQueryOptions = (projectId: string, media: Media) =>
    queryOptions({
        queryKey: getAnnotationsQueryKey(projectId, media),
        queryFn: () => annotationsQueryFn(projectId, media),
    });

const annotationsQueryFn = async (
    projectId: string,
    media: Media
): Promise<{ annotations: AnnotationDTO[]; user_reviewed: boolean; subset: DatasetSubset } | undefined> => {
    const queryParams = getAnnotationsQueryParams(media);

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
        return { annotations: [], user_reviewed: false, subset: 'unassigned' };
    }

    if (error) {
        throw error;
    }

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
            subset: data.subset,
        };
    }

    return data;
};

export const useAnnotationsQuery = (media: Media) => {
    const projectId = useProjectIdentifier();

    return useQuery(annotationsQueryOptions(projectId, media));
};
