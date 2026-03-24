// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import { useQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { range } from 'lodash-es';

import type { AnnotationDTO, Media, PredictionDTO } from '../../../constants/shared-types';
import { isVideoFrame } from '../../../shared/media-item-utils';
import { loadImageQueryOptions } from '../../annotator/hooks/use-load-image-query.hook';
import { useVideoPlayerContext } from '../../annotator/video-player/video-player-provider.component';
import { annotationsQueryOptions } from './api/use-annotations-query';

export const getInitialAnnotations = (isUserReviewed: boolean, annotationsDTO: AnnotationDTO[]): AnnotationDTO[] => {
    return isUserReviewed ? annotationsDTO : [];
};

export const getInitialPredictions = (predictions: PredictionDTO[] | undefined): AnnotationDTO[] => {
    return predictions ?? [];
};

export const getNextMediaItem = (currentMediaItem: Media, allMediaItems: Media[], step: number): Media | undefined => {
    if (isVideoFrame(currentMediaItem)) {
        const videoFrames = range(0, currentMediaItem.frame_count, step);
        const currentIndex = videoFrames.findIndex((frame) => frame === currentMediaItem.frame_number);

        if (currentIndex >= 0 && currentIndex < videoFrames.length - 1) {
            return {
                ...currentMediaItem,
                frame_number: videoFrames[currentIndex + 1],
            };
        } else {
            const nextFrame = videoFrames.find((frame) => frame > currentMediaItem.frame_number);

            if (nextFrame !== undefined) {
                return {
                    ...currentMediaItem,
                    frame_number: nextFrame ?? 0,
                };
            }
        }
    }

    const currentIndex = allMediaItems.findIndex(({ id }) => id === currentMediaItem.id);

    if (currentIndex < 0) {
        return allMediaItems[0];
    }

    if (currentIndex >= allMediaItems.length - 1) {
        return undefined;
    }

    return allMediaItems[currentIndex + 1];
};

export const useNextMediaItem = (currentMediaItem: Media, allMediaItems: Media[]) => {
    const context = useVideoPlayerContext();
    const step = context?.step ?? 1;

    return useMemo(() => {
        return getNextMediaItem(currentMediaItem, allMediaItems, step);
    }, [allMediaItems, currentMediaItem, step]);
};

// When the user navigates to next media, image data and annotations will be already in React Query cache,
// so the UI will feel smoother whenever the user switches image unless the user changes to a random or item.
// We could also consider those cases but I feel like it's overkill.
// Let's see how this improvement performs and then we can iterate on it.
//
// We trigger next-item data prefetch through disabled/conditional query hooks,
// so data is resolved from cache when available and fetched when needed.
export const useNextMediaPrefetch = (currentMediaItem: Media, allMediaItems: Media[]) => {
    const projectId = useProjectIdentifier();
    const nextMediaItem = useNextMediaItem(currentMediaItem, allMediaItems);

    const nextImageQuery = useQuery({
        ...loadImageQueryOptions(projectId, nextMediaItem ?? currentMediaItem),
        enabled: nextMediaItem !== undefined,
    });

    useQuery({
        ...annotationsQueryOptions(projectId, nextMediaItem ?? currentMediaItem),
        enabled: nextMediaItem !== undefined,
    });

    return {
        nextMediaItem,
        nextImage: nextImageQuery.data,
        isNextImageReady: nextImageQuery.isSuccess,
    };
};
