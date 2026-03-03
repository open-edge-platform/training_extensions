// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { RefObject, useEffect } from 'react';

import { useVideoPlayer } from './video-player-provider.component';

type VideoFrameProps = {
    canvasRef: RefObject<HTMLCanvasElement | null>;
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

        const updateCanvas: VideoFrameRequestCallback = () => {
            const ctx = canvasRef.current?.getContext('2d');

            if (videoRef.current === null || ctx == null) {
                return;
            }

            ctx.drawImage(video, 0, 0);

            callbackId = video.requestVideoFrameCallback(updateCanvas);

            const nextFrameIndex = Math.ceil(video.currentTime * videoFrame.fps);

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

export const VideoFrame = ({ canvasRef }: VideoFrameProps) => {
    const { videoRef } = useVideoPlayer();

    useRequestVideoFrameCallback(videoRef, canvasRef);

    return <></>;
};
