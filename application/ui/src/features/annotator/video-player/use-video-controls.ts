// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { RefObject, useState } from 'react';

import type { MediaVideo } from '../../../constants/shared-types';

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
    mediaItem: MediaVideo | undefined
): VideoControls => {
    const [isPlaying, setIsPlaying] = useState<boolean>(false);

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
    };

    const frames = mediaItem?.frame_count ?? 1;
    const fps = mediaItem?.fps || 1;
    const step = 1 / fps;
    const currentTime = videoRef.current?.currentTime ?? 0;

    // TODO: These will change once API for video frames is supported
    const canSelectPreviousFrame = currentTime - step >= 0;
    const canSelectNextFrame = currentTime + step <= frames / fps;

    const nextFrame = () => {
        if (!canSelectNextFrame || videoRef.current === null) {
            return;
        }

        videoRef.current.currentTime = videoRef.current.currentTime + step;
    };

    const previousFrame = () => {
        if (!canSelectPreviousFrame || videoRef.current === null) {
            return;
        }

        videoRef.current.currentTime = videoRef.current.currentTime - step;
    };

    const goto = (_frameNumber: number) => {
        //
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
