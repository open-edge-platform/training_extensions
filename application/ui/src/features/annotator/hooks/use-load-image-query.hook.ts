// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { Media } from '../../../constants/shared-types';
import { isVideoFrame } from '../../../shared/media-item-utils';
import { getMediaBinaryUrl, getVideoFrameBinaryUrl } from '../../../shared/media-url.utils';
import { getImageData, loadImage } from '../tools/utils';

export const useLoadImageQuery = (media: Media) => {
    const projectId = useProjectIdentifier();

    const queryKey = isVideoFrame(media)
        ? ['mediaItem', projectId, media.id, media.frame_number]
        : ['mediaItem', projectId, media.id];

    return useQuery({
        queryKey,
        queryFn: async () => {
            const url = isVideoFrame(media)
                ? getVideoFrameBinaryUrl(projectId, media.id, media.frame_number)
                : getMediaBinaryUrl(projectId, media.id);

            const image = await loadImage(url);

            return getImageData(image);
        },
        placeholderData: keepPreviousData,
        // The image of a media item never changes so we don't want to refetch stale data
        staleTime: Infinity,
        retry: 0,
    });
};
