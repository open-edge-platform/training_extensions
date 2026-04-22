// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useRef } from 'react';

import { useQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isEmpty, range } from 'lodash-es';
import { useLocalStorage } from 'usehooks-ts';

import type { AnnotationDTO, Media, PredictionDTO } from '../../../constants/shared-types';
import type { AnnotatorMode } from '../../../shared/annotator/annotator-mode';
import { isVideoFrame } from '../../../shared/media-item-utils';
import { loadImageQueryOptions } from '../../annotator/hooks/use-load-image-query.hook';
import { useVideoPlayerContext } from '../../annotator/video-player/video-player-provider.component';
import { annotationsQueryOptions } from './api/use-annotations-query';

export const getInitialAnnotations = (isUserReviewed: boolean, annotationsDTO: AnnotationDTO[]): AnnotationDTO[] => {
    return isUserReviewed ? annotationsDTO : [];
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

export const useAnnotatorMode = () => {
    const projectId = useProjectIdentifier();

    const [mode, setMode] = useLocalStorage<AnnotatorMode>(`${projectId}-annotator-mode`, 'annotation');

    return [mode, setMode] as const;
};

export const usePlayPauseVideoBySystem = (isLoadingPredictions: boolean) => {
    const isPausedBySystem = useRef<boolean>(false);
    const context = useVideoPlayerContext();

    const playRef = useRef(context?.videoControls.play);
    const pauseRef = useRef(context?.videoControls.pause);

    useEffect(() => {
        playRef.current = context?.videoControls.play;
    }, [context?.videoControls.play]);

    useEffect(() => {
        pauseRef.current = context?.videoControls.pause;
    }, [context?.videoControls.pause]);

    useEffect(() => {
        if (isLoadingPredictions && context?.videoControls.isPlaying) {
            isPausedBySystem.current = true;
            pauseRef.current?.();
        } else if (!isLoadingPredictions && isPausedBySystem.current) {
            isPausedBySystem.current = false;
            playRef.current?.();
        }
    }, [isLoadingPredictions, context?.videoControls.isPlaying]);
};
