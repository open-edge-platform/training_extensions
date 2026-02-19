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

    playbackRate: number;
    changePlaybackRate: (rate: number) => void;
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
    const [playbackRate, setPlaybackRate] = useState<number>(1);

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

    const toggleMute = () => {
        setIsMuted((prevIsMuted) => {
            const nextIsMuted = !prevIsMuted;

            if (videoRef.current === null) {
                return nextIsMuted;
            }

            videoRef.current.muted = nextIsMuted;

            return nextIsMuted;
        });
    };

    const frames = videoFrame?.frame_count ?? 1;
    const fps = videoFrame?.fps || 1;
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

    const changePlaybackRate = (rate: number) => {
        const previousRate = playbackRate;
        if (videoRef.current === null) {
            return;
        }

        try {
            setPlaybackRate(rate);
            videoRef.current.playbackRate = rate;
        } catch {
            setPlaybackRate(previousRate);
        }
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

                playbackRate,
                changePlaybackRate,
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
