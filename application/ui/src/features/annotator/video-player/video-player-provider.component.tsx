// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, RefObject, use, useRef } from 'react';

type VideoPlayerContextProps = {
    videoRef: RefObject<HTMLVideoElement | null>;
};

const VideoPlayerContext = createContext<VideoPlayerContextProps | null>(null);

type VideoPlayerProviderProps = {
    children: ReactNode;
};

export const VideoPlayerProvider = ({ children }: VideoPlayerProviderProps) => {
    const videoRef = useRef<HTMLVideoElement>(null);

    return <VideoPlayerContext value={{ videoRef }}>{children}</VideoPlayerContext>;
};

export const useVideoPlayer = () => {
    const context = use(VideoPlayerContext);

    if (context === null) {
        throw new Error('useVideoPlayer must be used within a VideoPlayerProvider');
    }
    return context;
};
