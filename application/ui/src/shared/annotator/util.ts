// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { InfiniteData, QueryClient } from '@tanstack/react-query';

import { Media, MediaWithPagination } from '../../constants/shared-types';
import { isVideo } from '../media-item-utils';

export const incrementCachedAnnotatedFrameCount = (queryClient: QueryClient, mediaItem: Media) => {
    queryClient.setQueriesData<InfiniteData<MediaWithPagination>>(
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
