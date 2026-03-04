// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback } from 'react';

import { range } from 'lodash-es';

import { Label, Media } from '../../../../constants/shared-types';
import { isVideoFrame } from '../../../../shared/media-item-utils';
import { useVideoPlayerContext } from '../../../annotator/video-player/video-player-provider.component';

export const toggleLabel = (newLabel: Label, labels: Label[]): Label[] => {
    const isExistingLabel = labels.some(({ id }) => id === newLabel.id);

    if (isExistingLabel) {
        return labels.filter(({ id }) => id !== newLabel.id) as Label[];
    }

    return [...labels, newLabel];
};

export const getNextItem = (totalItems: number, newIndex: number) => {
    return Math.min(totalItems, newIndex + 1);
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

export const useNextMedia = (currentMediaItem: Media, allMediaItems: Media[]) => {
    const context = useVideoPlayerContext();
    const step = context?.step ?? 1;

    return useCallback(() => {
        return getNextMediaItem(currentMediaItem, allMediaItems, step);
    }, [allMediaItems, currentMediaItem, step]);
};
