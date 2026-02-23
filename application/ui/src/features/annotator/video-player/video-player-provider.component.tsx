// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, RefObject, use, useRef, useState } from 'react';

import type { MediaVideo } from '../../../constants/shared-types';
import { isVideo, isVideoFrame } from '../../../shared/media-item-utils';
import { useVideoControls, VideoControls } from './use-video-controls';

type VideoPlayerContextProps = {
    videoRef: RefObject<HTMLVideoElement | null>;

    isMuted: boolean;
    toggleMute: () => void;

    videoFrame: MediaVideo;

    playbackRate: number;
    changePlaybackRate: (rate: number) => void;

    videoControls: VideoControls;
};

const VideoPlayerContext = createContext<VideoPlayerContextProps | null>(null);

type VideoPlayerProviderProps = {
    children: ReactNode;
    mediaItem: MediaVideo | undefined;
};

export const VideoPlayerProvider = ({ children, mediaItem }: VideoPlayerProviderProps) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const [isMuted, setIsMuted] = useState<boolean>(false);
    const [playbackRate, setPlaybackRate] = useState<number>(1);

    const videoControls = useVideoControls(videoRef, mediaItem);

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

    const value =
        mediaItem !== undefined
            ? {
                  videoFrame: mediaItem,
                  videoRef,
                  videoControls,

                  toggleMute,
                  isMuted,

                  playbackRate,
                  changePlaybackRate,
              }
            : null;

    return <VideoPlayerContext value={value}>{children}</VideoPlayerContext>;
};

export const useVideoPlayer = () => {
    const context = use(VideoPlayerContext);

    if (context === null) {
        throw new Error('useVideoPlayer must be used within a VideoPlayerProvider');
    }
    return context;
};

export const useVideoPlayerContext = () => {
    return use(VideoPlayerContext);
};
