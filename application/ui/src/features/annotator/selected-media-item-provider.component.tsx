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

// See tools/utils.ts for the oversized image handling flow (downscale → display
// in media-space → disable smart tools or coordinate-transform at boundaries).
type SelectedMediaItemContextProps = {
    mediaItem: Media;
    roi: RegionOfInterest;
    setMediaItem: (item: Media) => void;
    // Media-space view of the image: width/height always match the media item so
    // coordinate clamping in drawing tools stays in annotation space.
    image: ImageData;
    // True once the current item's full-resolution pixels are loaded and decoded.
    // False during loading, while a placeholder is shown, or for images too large
    // to rasterise at full size (smart tools are disabled in that case).
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

    const {
        data: loadedImage = getImageData(new Image()),
        isSuccess,
        isPlaceholderData,
    } = useLoadImageQuery(mediaItem);

    // For oversized media: wrap downscaled data in full-size dimensions so
    // drawing tools clamp to media-space, not the smaller buffer dimensions.
    const decodedAtFullSize = loadedImage.width === mediaItem.width && loadedImage.height === mediaItem.height;
    const image = decodedAtFullSize
        ? loadedImage
        : ({
              width: mediaItem.width,
              height: mediaItem.height,
              data: loadedImage.data,
              colorSpace: 'srgb',
          } as ImageData);
    const isImageReady = isSuccess && !isPlaceholderData && decodedAtFullSize;

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
