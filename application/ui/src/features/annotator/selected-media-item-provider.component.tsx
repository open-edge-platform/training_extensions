// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useEffect, useState } from 'react';

import type { Media } from '../../constants/shared-types';
import { isVideo } from '../../shared/media-item-utils';
import type { RegionOfInterest } from '../../shared/types';
import { useLoadImageQuery } from './hooks/use-load-image-query.hook';

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

export const SelectedMediaItemProvider = ({
    mediaItem: initialMediaItem,
    children,
}: SelectedMediaItemProviderProps) => {
    const [mediaItem, setMediaItem] = useState<Media>(initialMediaItem);

    /* TODO: Check if we need this */
    useEffect(() => {
        setMediaItem(initialMediaItem);
    }, [initialMediaItem]);

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

    const mediaItemId = isVideo(mediaItem) ? undefined : mediaItem.id;

    const { data: image } = useLoadImageQuery(mediaItemId);

    return <MediaItemImageContext value={{ image }}>{children}</MediaItemImageContext>;
};

export const useMediaItemImage = (): MediaItemImageContextType => {
    const context = useContext(MediaItemImageContext);

    if (context === null) {
        throw new Error('useMediaItemImage was used outside of MediaItemImageLoader');
    }

    return context;
};
