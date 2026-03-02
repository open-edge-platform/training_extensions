// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { Media } from '../../../constants/shared-types';
import { isVideoFrame } from '../../../shared/media-item-utils';
import { getMediaBinaryUrl, getVideoFrameBinaryUrl } from '../../../shared/media-url.utils';
import { getImageData, loadImage } from '../tools/utils';

export const getLoadImageQueryKey = (projectId: string, media: Media) => {
    return isVideoFrame(media)
        ? ['mediaItem', projectId, media.id, media.frame_number]
        : ['mediaItem', projectId, media.id];
};

export const loadImageQueryFn = async (projectId: string, media: Media) => {
    const url = isVideoFrame(media)
        ? getVideoFrameBinaryUrl(projectId, media.id, media.frame_number)
        : getMediaBinaryUrl(projectId, media.id);

    const image = await loadImage(url);

    return getImageData(image);
};

export const useLoadImageQuery = (media: Media) => {
    const projectId = useProjectIdentifier();

    return useQuery({
        queryKey: getLoadImageQueryKey(projectId, media),
        queryFn: () => loadImageQueryFn(projectId, media),
        placeholderData: keepPreviousData,
        // The image of a media item never changes so we don't want to refetch stale data
        staleTime: Infinity,
        retry: 0,
    });
};
