// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, RefObject, use, useRef, useState } from 'react';

import { type Media } from '../../../constants/shared-types';

type VideoPlayerContextProps = {
    videoRef: RefObject<HTMLVideoElement | null>;

    play: () => Promise<void>;
    pause: () => void;
    isPlaying: boolean;

    isMuted: boolean;
    toggleMute: () => void;

    nextFrame: () => void;
    canSelectNextFrame: boolean;

    previousFrame: () => void;
    canSelectPreviousFrame: boolean;

    videoFrame: Media | undefined;
};

const VideoPlayerContext = createContext<VideoPlayerContextProps | null>(null);

type VideoPlayerProviderProps = {
    children: ReactNode;
    videoFrame: Media | undefined;
};

export const VideoPlayerProvider = ({ children, videoFrame }: VideoPlayerProviderProps) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const [isPlaying, setIsPlaying] = useState<boolean>(false);
    const [isMuted, setIsMuted] = useState<boolean>(false);

    const play = async () => {
        if (videoRef.current === null) {
            return;
        }
        setIsPlaying(true);
        await videoRef.current.play();
    };

    const pause = () => {
        if (videoRef.current === null) {
            return;
        }
        setIsPlaying(false);
        videoRef.current.pause();
    };

    const toggleMute = () => {
        if (videoRef.current === null) {
            return;
        }
        setIsMuted(!isMuted);
        videoRef.current.muted = !isMuted;
    };

    const frames = videoFrame?.frame_count ?? 1;
    const fps = videoFrame?.fps ?? 1;
    const step = (frames / fps) * 0.01;
    const currentTime = videoRef.current?.currentTime ?? 0;

    // TODO: These will change once API for video frames is supported
    const canSelectPreviousFrame = currentTime - step >= 0;
    const canSelectNextFrame = currentTime + step <= frames * fps;

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

    return (
        <VideoPlayerContext
            value={{
                videoFrame,
                videoRef,
                isPlaying,
                play,
                pause,
                toggleMute,
                isMuted,
                nextFrame,
                previousFrame,
                canSelectNextFrame,
                canSelectPreviousFrame,
            }}
        >
            {children}
        </VideoPlayerContext>
    );
};

export const useVideoPlayer = () => {
    const context = use(VideoPlayerContext);

    if (context === null) {
        throw new Error('useVideoPlayer must be used within a VideoPlayerProvider');
    }
    return context;
};
