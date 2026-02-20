// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { RefObject, useEffect } from 'react';

import { VisuallyHidden } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { type Media } from '../../../constants/shared-types';
import { getMediaBinaryUrl } from '../../../shared/media-url.utils';
import { useVideoPlayer } from './video-player-provider.component';

type VideoFrameProps = {
    canvasRef: RefObject<HTMLCanvasElement | null>;
    mediaItem: Media;
};

const useRequestVideoFrameCallback = (
    videoRef: RefObject<HTMLVideoElement | null>,
    canvasRef: RefObject<HTMLCanvasElement | null>
) => {
    const { videoControls } = useVideoPlayer();
    const { isPlaying } = videoControls;

    useEffect(() => {
        if (!isPlaying) {
            return;
        }

        if (videoRef.current === null) {
            return;
        }

        const video = videoRef.current;

        let callbackId: number | null = null;

        const updateCanvas: VideoFrameRequestCallback = () => {
            if (videoRef.current === null) {
                return;
            }

            const ctx = canvasRef.current?.getContext('2d');

            if (ctx == null) {
                return;
            }

            ctx.drawImage(video, 0, 0);

            callbackId = video.requestVideoFrameCallback(updateCanvas);
        };

        callbackId = video.requestVideoFrameCallback(updateCanvas);

        return () => {
            if (callbackId !== null) {
                video.cancelVideoFrameCallback(callbackId);
            }
        };
    }, [videoRef, canvasRef, isPlaying]);

    return videoRef;
};

export const VideoFrame = ({ canvasRef, mediaItem }: VideoFrameProps) => {
    const projectId = useProjectIdentifier();
    const { videoRef } = useVideoPlayer();

    useRequestVideoFrameCallback(videoRef, canvasRef);

    return (
        <VisuallyHidden>
            <video
                ref={videoRef}
                src={getMediaBinaryUrl(projectId, mediaItem.id)}
                width={mediaItem.width}
                height={mediaItem.height}
                preload={'auto'}
                muted
            />
        </VisuallyHidden>
    );
};
