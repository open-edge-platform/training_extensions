// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useRef, useState } from 'react';

import { isEqual } from 'lodash-es';
import { useParams } from 'react-router';

import type { Media } from '../../constants/shared-types';
import { isVideo } from '../../shared/media-item-utils';
import type { RegionOfInterest } from '../../shared/types';
import { useLoadImageQuery } from './hooks/use-load-image-query.hook';
import { getImageData } from './tools/utils';

type SelectedMediaItemContextProps = {
    mediaItem: Media;
    roi: RegionOfInterest;
    setMediaItem: (item: Media) => void;
};

const SelectedMediaItemContext = createContext<SelectedMediaItemContextProps | null>(null);

type SelectedMediaItemProviderProps = {
    mediaItem: Media;
    children: ReactNode;
};

const useVideoFrameNumberQueryParam = () => {
    const { frameNumber } = useParams<{ frameNumber?: string }>();

    return frameNumber ? parseInt(frameNumber) : 0;
};

const getMediaItem = (mediaItem: Media, frameNumber: number): Media => {
    if (isVideo(mediaItem)) {
        return {
            ...mediaItem,
            type: 'video_frame',
            frame_stride: mediaItem.fps,
            frame_number: frameNumber,
        };
    }

    return mediaItem;
};

const useMediaItem = (initialMediaItem: Media) => {
    const frameNumber = useVideoFrameNumberQueryParam();

    const [mediaItem, setMediaItem] = useState<Media>(() => getMediaItem(initialMediaItem, frameNumber));
    const prevInitialMediaItem = useRef(initialMediaItem);

    if (!isEqual(initialMediaItem, prevInitialMediaItem.current)) {
        prevInitialMediaItem.current = initialMediaItem;
        setMediaItem(getMediaItem(initialMediaItem, frameNumber));
    }

    return [mediaItem, setMediaItem] as const;
};

export const SelectedMediaItemProvider = ({
    mediaItem: initialMediaItem,
    children,
}: SelectedMediaItemProviderProps) => {
    const [mediaItem, setMediaItem] = useMediaItem(initialMediaItem);

    const roi: RegionOfInterest = { x: 0, y: 0, width: mediaItem.width, height: mediaItem.height };

    return <SelectedMediaItemContext value={{ mediaItem, roi, setMediaItem }}>{children}</SelectedMediaItemContext>;
};

export const useSelectedMediaItem = (): SelectedMediaItemContextProps => {
    const context = useContext(SelectedMediaItemContext);

    if (context === null) {
        throw new Error('useSelectedMediaItem was used outside of SelectedMediaItemProvider');
    }

    return context;
};

type MediaItemImageContextType = {
    image: ImageData;
};

const MediaItemImageContext = createContext<MediaItemImageContextType | null>(null);

/*
TODO: Use this when API supports video frames
const getMediaItem = (mediaItem: Media) => {
    if (isVideo(mediaItem)) {
        // For video, we want to get the first frame of the video, that's not supported right now
        return undefined;
    }

    return mediaItem;
};*/

/**
 * Loads the image for the currently selected media item via a suspense query.
 * Must be placed INSIDE a <Suspense> boundary so that only the canvas area
 * suspends — not the toolbars or annotation state above it.
 */
export const MediaItemImageLoader = ({ children }: { children: ReactNode }) => {
    const { mediaItem } = useSelectedMediaItem();

    const { data: image = getImageData(new Image()) } = useLoadImageQuery(mediaItem);

    return <MediaItemImageContext value={{ image }}>{children}</MediaItemImageContext>;
};

export const useMediaItemImage = (): MediaItemImageContextType => {
    const context = useContext(MediaItemImageContext);

    if (context === null) {
        throw new Error('useMediaItemImage was used outside of MediaItemImageLoader');
    }

    return context;
};
