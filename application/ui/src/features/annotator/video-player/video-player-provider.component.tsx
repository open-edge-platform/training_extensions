// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, RefObject, use, useRef, useState } from 'react';

type VideoPlayerContextProps = {
    videoRef: RefObject<HTMLVideoElement | null>;

    play: () => Promise<void>;
    pause: () => void;
    isPlaying: boolean;

    isMuted: boolean;
    toggleMute: () => void;
};

const VideoPlayerContext = createContext<VideoPlayerContextProps | null>(null);

type VideoPlayerProviderProps = {
    children: ReactNode;
};

export const VideoPlayerProvider = ({ children }: VideoPlayerProviderProps) => {
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

    return (
        <VideoPlayerContext value={{ videoRef, isPlaying, play, pause, toggleMute, isMuted }}>
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
