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
    image: ImageData;
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

    const { data: image = getImageData(new Image()) } = useLoadImageQuery(mediaItem);

    const roi: RegionOfInterest = { x: 0, y: 0, width: mediaItem.width, height: mediaItem.height };

    return (
        <SelectedMediaItemContext value={{ mediaItem, roi, setMediaItem, image }}>{children}</SelectedMediaItemContext>
    );
};

export const useSelectedMediaItem = (): SelectedMediaItemContextProps => {
    const context = useContext(SelectedMediaItemContext);

    if (context === null) {
        throw new Error('useSelectedMediaItem was used outside of SelectedMediaItemProvider');
    }

    return context;
};
