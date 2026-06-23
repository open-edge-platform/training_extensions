// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef } from 'react';

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import type { Media } from '../../../constants/shared-types';
import { isVideo, isVideoFrame } from '../../../shared/media-item-utils';
import { getMediaBinaryUrl } from '../../../shared/media-url.utils';
import { VideoFrame } from '../video-player/video-frame.component';
import { drawImageDataOnCanvas } from './draw-image-data-on-canvas';

import classes from './annotator-canvas.module.scss';

const useDrawImageOnCanvas = (image: ImageData) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const ctx = canvasRef?.current?.getContext('2d');

        if (ctx == null) {
            return;
        }

        drawImageDataOnCanvas(ctx, image);
    }, [image]);

    return canvasRef;
};

// Oversized static images: use <img> to bypass canvas rasterization limit.
const StaticImage = ({ mediaItem }: { mediaItem: Media }) => {
    const projectId = useProjectIdentifier();

    return (
        <img
            src={getMediaBinaryUrl(projectId, mediaItem.id)}
            width={mediaItem.width}
            height={mediaItem.height}
            crossOrigin='anonymous'
            className={classes.image}
            alt=''
        />
    );
};

type CanvasMediaImageProps = {
    image: ImageData;
    showVideoFrame?: boolean;
};

const CanvasMediaImage = ({ image, showVideoFrame = false }: CanvasMediaImageProps) => {
    const canvasRef = useDrawImageOnCanvas(image);

    return (
        <>
            <canvas ref={canvasRef} width={image.width} height={image.height} className={classes.image} />
            {showVideoFrame && <VideoFrame canvasRef={canvasRef} />}
        </>
    );
};

type MediaImageProps = {
    image: ImageData;
    mediaItem: Media;
};

export const MediaImage = ({ image, mediaItem }: MediaImageProps) => {
    if (isVideo(mediaItem) || isVideoFrame(mediaItem)) {
        return <CanvasMediaImage image={image} showVideoFrame />;
    }

    // Render via canvas only when decoded at native resolution.
    // `ImageData.data.length` is expected to be width * height * 4
    // (RGBA: 4 channels, 1 byte per channel = 4 bytes per pixel).
    const isFullResolution =
        image.width === mediaItem.width &&
        image.height === mediaItem.height &&
        image.data.length === mediaItem.width * mediaItem.height * 4;

    if (isFullResolution) {
        return <CanvasMediaImage image={image} />;
    }

    return <StaticImage mediaItem={mediaItem} />;
};
