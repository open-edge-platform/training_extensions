// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQuery, UseQueryResult } from '@tanstack/react-query';

import { API_BASE_URL } from '../../../api/client';
import { useProjectIdentifier } from '../../../hooks/use-project-identifier.hook';
import { getImageData, loadImage } from '../tools/utils';
import { DatasetItem } from '../types';

export const useLoadImageQuery = (mediaItem: DatasetItem | undefined): UseQueryResult<ImageData, unknown> => {
    const projectId = useProjectIdentifier();

    return useQuery({
        queryKey: ['mediaItem', mediaItem?.id, projectId],
        queryFn: async () => {
            if (mediaItem === undefined) {
                throw new Error("Can't fetch undefined media item");
            }

            const imageUrl = `${API_BASE_URL}/api/projects/${projectId}/dataset/items/${mediaItem.id}/binary`;
            const image = await loadImage(imageUrl);

            return getImageData(image);
        },
        enabled: mediaItem !== undefined && Boolean(projectId),
        // The image of a media item never changes so we don't want to refetch stale data
        staleTime: Infinity,
        retry: 0,
    });
};
