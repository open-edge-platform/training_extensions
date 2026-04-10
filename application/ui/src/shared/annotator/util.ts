// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { InfiniteData, QueryClient } from '@tanstack/react-query';

import { SchemaMediaWithPagination } from '../../api/openapi-spec';
import { Media } from '../../constants/shared-types';
import { isVideo } from '../media-item-utils';

/* const updateAnnotatedFrameCount = (mediaItem: Media) => (item: MediaDTO) => {
    return isVideo(item) && item.id === mediaItem.id
        ? { ...item, annotated_frame_count: item.annotated_frame_count + 1 }
        : item;
}; */

export const incrementCachedAnnotatedFrameCount = (queryClient: QueryClient, mediaItem: Media) => {
    queryClient.setQueriesData<InfiniteData<SchemaMediaWithPagination>>(
        { queryKey: ['get', '/api/projects/{project_id}/dataset/media'] },
        (oldData) => {
            if (!oldData?.pages) return oldData;

            return {
                ...oldData,
                pages: oldData.pages.map((page) => ({
                    ...page,
                    items: page.items.map((item) =>
                        isVideo(item) && item.id === mediaItem.id
                            ? { ...item, annotated_frame_count: item.annotated_frame_count + 1 }
                            : item
                    ),
                })),
            };
        }
    );
};
