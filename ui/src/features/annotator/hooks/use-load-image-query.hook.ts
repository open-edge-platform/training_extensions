// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQuery, UseQueryResult } from '@tanstack/react-query';

import { getImageData, loadImage } from '../tools/utils';
import { MediaItem } from '../types';

export const useLoadImageQuery = (mediaItem: MediaItem | undefined): UseQueryResult<ImageData, unknown> => {
    return useQuery({
        queryKey: ['mediaItem', mediaItem?.id],
        queryFn: async () => {
            if (mediaItem === undefined) {
                throw new Error("Can't fetch undefined media item");
            }

            const image = await loadImage(mediaItem.thumbhash);

            return getImageData(image);
        },
        enabled: mediaItem !== undefined,
        // The image of a media item never changes so we don't want to refetch stale data
        staleTime: Infinity,
        retry: 0,
    });
};
