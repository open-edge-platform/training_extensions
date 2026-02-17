// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useSuspenseQuery, UseSuspenseQueryResult } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { API_BASE_URL } from '../../../api/client';
import { getImageData, loadImage } from '../tools/utils';

export const useLoadImageQuery = (mediaItemId: string | undefined): UseSuspenseQueryResult<ImageData, unknown> => {
    const projectId = useProjectIdentifier();

    return useSuspenseQuery({
        queryKey: ['mediaItem', mediaItemId, projectId],
        queryFn: async () => {
            if (mediaItemId === undefined) {
                throw new Error("Can't fetch undefined media item");
            }

            const imageUrl = `${API_BASE_URL}/api/projects/${projectId}/dataset/media/${mediaItemId}/binary`;
            const image = await loadImage(imageUrl);

            return getImageData(image);
        },
        // The image of a media item never changes so we don't want to refetch stale data
        staleTime: Infinity,
        retry: 0,
    });
};
