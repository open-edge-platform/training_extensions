// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useCallback, useContext, useRef, useState } from 'react';

import { isEqual } from 'lodash-es';
import { useParams } from 'react-router';

import type { Media } from '../../constants/shared-types';
import { isVideo, isVideoFrame } from '../../shared/media-item-utils';
import type { RegionOfInterest } from '../../shared/types';
import { useLoadImageQuery } from './hooks/use-load-image-query.hook';
import { getImageData } from './tools/utils';

type SelectedMediaItemContextProps = {
    mediaItem: Media;
    roi: RegionOfInterest;
    setMediaItem: (item: Media) => void;
    image: ImageData;
    isImageReady: boolean;
};

const SelectedMediaItemContext = createContext<SelectedMediaItemContextProps | null>(null);

type SelectedMediaItemProviderProps = {
    mediaItem: Media;
    children: ReactNode;
};

const useVideoFrameNumberPathParam = () => {
    const { frameNumber } = useParams<{ frameNumber?: string }>();

    if (frameNumber === undefined) {
        return 0;
    }

    const frameNumberInt = parseInt(frameNumber, 10);

    return Number.isNaN(frameNumberInt) ? 0 : frameNumberInt;
};

const convertMediaItem = (mediaItem: Media, frameNumber: number): Media => {
    if (isVideo(mediaItem)) {
        return {
            ...mediaItem,
            type: 'video_frame',
            fps: Math.round(mediaItem.fps),
            frame_stride: Math.round(mediaItem.fps),
            frame_number: frameNumber,
        };
    }

    return mediaItem;
};

const useMediaItem = (initialMediaItem: Media) => {
    const frameNumberFromPathParam = useVideoFrameNumberPathParam();
    const maxFrameNumber =
        isVideo(initialMediaItem) || isVideoFrame(initialMediaItem) ? initialMediaItem.frame_count - 1 : 0;
    const frameNumber = Math.max(0, Math.min(frameNumberFromPathParam, maxFrameNumber));

    const [mediaItem, setMediaItem] = useState<Media>(() => convertMediaItem(initialMediaItem, frameNumber));
    const prevInitialMediaItem = useRef(initialMediaItem);

    if (initialMediaItem.id !== prevInitialMediaItem.current.id) {
        prevInitialMediaItem.current = initialMediaItem;
        setMediaItem(convertMediaItem(initialMediaItem, frameNumber));
    }

    const changeMediaItem = useCallback((newMedia: Media) => {
        setMediaItem((prevMediaItem) => {
            if (isEqual(prevMediaItem, newMedia)) {
                return prevMediaItem;
            }

            return convertMediaItem(newMedia, 0);
        });
    }, []);

    return [mediaItem, changeMediaItem] as const;
};

export const SelectedMediaItemProvider = ({
    mediaItem: initialMediaItem,
    children,
}: SelectedMediaItemProviderProps) => {
    const [mediaItem, setMediaItem] = useMediaItem(initialMediaItem);

    const { data: image = getImageData(new Image()), isSuccess, isPlaceholderData } = useLoadImageQuery(mediaItem);
    const isImageReady = isSuccess && !isPlaceholderData;

    const roi: RegionOfInterest = { x: 0, y: 0, width: mediaItem.width, height: mediaItem.height };

    return (
        <SelectedMediaItemContext value={{ mediaItem, roi, setMediaItem, image, isImageReady }}>
            {children}
        </SelectedMediaItemContext>
    );
};

export const useSelectedMediaItem = (): SelectedMediaItemContextProps => {
    const context = useContext(SelectedMediaItemContext);

    if (context === null) {
        throw new Error('useSelectedMediaItem was used outside of SelectedMediaItemProvider');
    }

    return context;
};
