// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { keepPreviousData, queryOptions, useQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { Media } from '../../../constants/shared-types';
import { isVideoFrame } from '../../../shared/media-item-utils';
import { getMediaBinaryUrl, getVideoFrameBinaryUrl } from '../../../shared/media-url.utils';
import { getImageData, loadImage } from '../tools/utils';

export const loadImageQueryOptions = (projectId: string, media: Media) =>
    queryOptions({
        queryKey: isVideoFrame(media)
            ? ['mediaItem', projectId, media.id, media.frame_number]
            : ['mediaItem', projectId, media.id],
        queryFn: async () => {
            const url = isVideoFrame(media)
                ? getVideoFrameBinaryUrl(projectId, media.id, media.frame_number)
                : getMediaBinaryUrl(projectId, media.id);

            const image = await loadImage(url);

            return getImageData(image);
        },
        staleTime: Infinity,
        retry: 0,
    });

export const useLoadImageQuery = (media: Media) => {
    const projectId = useProjectIdentifier();

    return useQuery({
        ...loadImageQueryOptions(projectId, media),
        placeholderData: keepPreviousData,
    });
};
