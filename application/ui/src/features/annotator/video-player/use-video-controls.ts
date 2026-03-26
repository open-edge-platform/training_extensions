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

const roundFrameNumber = (frameNumber: number, step: number) => Math.round(frameNumber / step) * step;

export const useVideoControls = ({
    step,
    videoRef,
    videoFrame,
    selectVideoFrame,
    changeCurrentFrameIndex,
}: {
    step: number;
    videoRef: RefObject<HTMLVideoElement | null>;
    videoFrame: MediaVideoFrame | undefined;
    selectVideoFrame: (media: MediaVideoFrame) => void;
    changeCurrentFrameIndex: (index: number) => void;
}): VideoControls => {
    const [isPlaying, setIsPlaying] = useState<boolean>(false);

    const totalFrames = videoFrame?.frame_count ?? 1;
    const currentFrameNumber = videoFrame?.frame_number ?? 0;

    const previousVideoFrameNumber = roundFrameNumber(currentFrameNumber - step, step);
    const canSelectPreviousFrame = previousVideoFrameNumber >= 0;

    const nextVideoFrameNumber = roundFrameNumber(currentFrameNumber + step, step);
    const canSelectNextFrame = nextVideoFrameNumber < totalFrames;

    const selectFrame = (frameNumber: number) => {
        if (videoRef.current === null || videoFrame === undefined) {
            return;
        }
        selectVideoFrame({ ...videoFrame, frame_number: frameNumber });
        changeCurrentFrameIndex(frameNumber);

        videoRef.current.currentTime = (frameNumber + 1) / videoFrame.fps;
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
        if (videoRef.current === null || videoFrame === undefined) {
            return;
        }

        setIsPlaying(false);
        videoRef.current.pause();

        const maxNearestFrame = Math.floor((videoFrame.frame_count - 1) / step) * step;
        const nearestFrame = Math.min(maxNearestFrame, roundFrameNumber(currentFrameNumber, step));

        goto(nearestFrame);
    };

    const nextFrame = () => {
        if (!canSelectNextFrame || videoRef.current === null) {
            return;
        }

        if (!isPlaying) {
            selectFrame(nextVideoFrameNumber);
        } else {
            changeCurrentFrameIndex(nextVideoFrameNumber);
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
            changeCurrentFrameIndex(previousVideoFrameNumber);
            videoRef.current.currentTime -= 1;
        }
    };

    const goto = (frameNumber: number) => {
        if (videoRef.current === null || videoFrame === undefined) {
            return;
        }

        if (frameNumber >= totalFrames || frameNumber < 0) {
            return;
        }

        if (isPlaying) {
            videoRef.current.pause();
            setIsPlaying(false);
        }

        const nearest = Math.min(roundFrameNumber(frameNumber, step), totalFrames - 1);

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
