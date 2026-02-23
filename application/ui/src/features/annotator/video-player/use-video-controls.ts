// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { RefObject, useState } from 'react';

import type { MediaVideoFrame } from '../../../constants/shared-types';

export type VideoControls = {
    canSelectPreviousFrame: boolean;
    previousFrame: () => void;
    canSelectNextFrame: boolean;
    isPlaying: boolean;
    nextFrame: () => void;
    play: () => Promise<void>;
    pause: () => void;
    goto: (frameNumber: number) => void;
    canPlay?: boolean;
};

export const useVideoControls = (
    videoRef: RefObject<HTMLVideoElement | null>,
    mediaItem: MediaVideoFrame | undefined,
    selectVideoFrame: (media: MediaVideoFrame) => void,
    changeCurrentFrameIndex: (index: number) => void
): VideoControls => {
    const [isPlaying, setIsPlaying] = useState<boolean>(false);

    const totalFrames = mediaItem?.frame_count ?? 1;
    const step = mediaItem?.frame_stride;
    const currentFrameNumber = mediaItem.frame_number;

    const round = (x: number) => Math.round(x / step) * step;
    const previousVideoFrameNumber = round(currentFrameNumber - step);
    const canSelectPreviousFrame = previousVideoFrameNumber >= 0;

    const nextVideoFrameNumber = round(currentFrameNumber + step);
    const canSelectNextFrame = nextVideoFrameNumber < totalFrames;

    const selectFrame = (frameNumber: number) => {
        if (videoRef.current === null) {
            return;
        }
        // TODO: Update selected video frame in selected media item provider
        selectVideoFrame({ ...mediaItem, frame_number: frameNumber });
        changeCurrentFrameIndex(frameNumber);

        videoRef.current.currentTime = (frameNumber + 1) / mediaItem.fps;
    };

    const play = async () => {
        if (videoRef.current === null) {
            return;
        }
        try {
            setIsPlaying(true);
            await videoRef.current.play();
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Unknown error';

            setIsPlaying(false);
            console.error(`Error while playing video: ${message}`);
        }
    };

    const pause = () => {
        if (videoRef.current === null) {
            return;
        }

        setIsPlaying(false);
        videoRef.current.pause();

        const maxNearestFrame = Math.floor((mediaItem.frame_count - 1) / step) * step;
        const nearestFrame = Math.min(maxNearestFrame, Math.round(currentFrameNumber / step) * step);

        goto(nearestFrame);
    };

    const nextFrame = () => {
        if (!canSelectNextFrame || videoRef.current === null) {
            return;
        }

        if (!isPlaying) {
            selectFrame(nextVideoFrameNumber);
        } else {
            videoRef.current.currentTime += 1;
        }
    };

    const previousFrame = () => {
        if (!canSelectPreviousFrame || videoRef.current === null) {
            return;
        }

        if (!isPlaying) {
            selectFrame(previousVideoFrameNumber);
        } else {
            videoRef.current.currentTime -= 1;
        }
    };

    const goto = (frameNumber: number) => {
        if (videoRef.current === null) {
            return;
        }

        if (frameNumber >= totalFrames || frameNumber < 0) {
            return;
        }

        if (isPlaying) {
            videoRef.current.pause();
            setIsPlaying(false);
        }

        videoRef.current.currentTime = (frameNumber + 1) / mediaItem.fps;
        const nearest = Math.min(Math.round(frameNumber / step) * step, totalFrames - 1);

        selectFrame(nearest);
    };

    return {
        isPlaying,
        pause,
        play,
        canSelectNextFrame,
        canSelectPreviousFrame,
        nextFrame,
        previousFrame,
        goto,
    };
};
