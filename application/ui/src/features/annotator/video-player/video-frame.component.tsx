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
    const { videoControls, changeCurrentFrameIndex, videoFrame } = useVideoPlayer();
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

        const updateCanvas: VideoFrameRequestCallback = (_now, metadata) => {
            if (videoRef.current === null) {
                return;
            }

            const ctx = canvasRef.current?.getContext('2d');

            if (ctx == null) {
                return;
            }

            ctx.drawImage(video, 0, 0);

            callbackId = video.requestVideoFrameCallback(updateCanvas);

            const nextFrameIndex = Math.ceil(metadata.mediaTime * videoFrame.fps);
            changeCurrentFrameIndex(nextFrameIndex);
        };

        callbackId = video.requestVideoFrameCallback(updateCanvas);

        return () => {
            if (callbackId !== null) {
                video.cancelVideoFrameCallback(callbackId);
            }
        };
    }, [videoRef, canvasRef, isPlaying, videoFrame.fps, changeCurrentFrameIndex]);

    return videoRef;
};

export const VideoFrame = ({ canvasRef, mediaItem }: VideoFrameProps) => {
    const projectId = useProjectIdentifier();
    const { videoRef, videoControls } = useVideoPlayer();

    useRequestVideoFrameCallback(videoRef, canvasRef);

    const handleEnded = () => {
        if (videoRef.current === null) {
            return;
        }
        videoControls.pause();
        videoRef.current.currentTime = 0;
    };

    return (
        <VisuallyHidden>
            <video
                ref={videoRef}
                src={getMediaBinaryUrl(projectId, mediaItem.id)}
                width={mediaItem.width}
                height={mediaItem.height}
                preload={'auto'}
                onEnded={handleEnded}
                muted
            />
        </VisuallyHidden>
    );
};
