// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../api/client';
import { DatasetItemAnnotationStatus, DatasetSubset, Media, MediaDTO, Pagination } from '../constants/shared-types';

const DATASET_ITEMS_LIMIT = 20;

interface UseGetDatasetMediaItemsOptions {
    subset?: DatasetSubset;
    annotationStatus?: DatasetItemAnnotationStatus;
    labelIds?: string[];
}

const getMediaEntities = (items: MediaDTO[]): Media[] => {
    return items.map((item) => {
        // We will never get the video frame using '/api/projects/{project_id}/dataset/media', it's added only because
        // of documentation reasons. We use MediaVideoFrame as a local type to work with the played frame in the video.
        if (item.type === 'video_frame') {
            return {
                duration: 0,
                frame_count: 0,
                annotated_frame_count: 0,
                fps: 0,
                frame_number: 0,
                frame_stride: 0,
                ...item,
            };
        }

        return item;
    });
};

export const useGetDatasetMediaItems = (options?: UseGetDatasetMediaItemsOptions) => {
    const project_id = useProjectIdentifier();

    const query: {
        offset: number;
        limit: number;
        subset?: DatasetSubset;
        annotation_status?: DatasetItemAnnotationStatus;
        labels?: string[];
    } = {
        offset: 0,
        limit: DATASET_ITEMS_LIMIT,
    };

    if (options?.subset !== undefined) {
        query.subset = options.subset;
    }

    if (options?.annotationStatus !== undefined) {
        query.annotation_status = options.annotationStatus;
    }

    if (options?.labelIds !== undefined) {
        query.labels = options.labelIds;
    }

    const { data, fetchNextPage, hasNextPage, isFetchingNextPage, isPending } = $api.useInfiniteQuery(
        'get',
        '/api/projects/{project_id}/dataset/media',
        { params: { query, path: { project_id } } },
        {
            pageParamName: 'offset',
            getNextPageParam: ({ pagination }: { pagination: Pagination }) => {
                const total = pagination.offset + pagination.count;

                if (total >= pagination.total) {
                    return undefined;
                }

                return pagination.offset + DATASET_ITEMS_LIMIT;
            },
        }
    );

    const items = useMemo(() => {
        const mediaItems = data?.pages.flatMap((page) => page.items) ?? [];

        return getMediaEntities(mediaItems);
    }, [data?.pages]);
    const totalCount = data?.pages[0]?.pagination?.total ?? 0;

    return { items, fetchNextPage, hasNextPage, isFetchingNextPage, isPending, totalCount };
};
