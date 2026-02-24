// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, RefObject, use, useMemo, useRef, useState } from 'react';

import type { MediaVideo, MediaVideoFrame } from '../../../constants/shared-types';
import { useVideoControls, VideoControls } from './use-video-controls';

type VideoPlayerContextProps = {
    videoRef: RefObject<HTMLVideoElement | null>;

    isMuted: boolean;
    toggleMute: () => void;

    videoFrame: MediaVideoFrame;

    playbackRate: number;
    changePlaybackRate: (rate: number) => void;

    videoControls: VideoControls;

    changeCurrentFrameIndex: (index: number) => void;
};

const VideoPlayerContext = createContext<VideoPlayerContextProps | null>(null);

type VideoPlayerProviderProps = {
    children: ReactNode;
    // TODO: Narrow the type to be MediaVideoFrame | undefined
    mediaItem: MediaVideo | MediaVideoFrame | undefined;
    changeSelectedMediaItem: (media: MediaVideoFrame) => void;
};

export const VideoPlayerProvider = ({ children, mediaItem, changeSelectedMediaItem }: VideoPlayerProviderProps) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const [isMuted, setIsMuted] = useState<boolean>(false);
    const [playbackRate, setPlaybackRate] = useState<number>(1);
    // TODO: Update default to be media item frame index
    const [currentFrameIndex, setCurrentFrameIndex] = useState<number>(0);

    const playingVideoFrame: MediaVideoFrame | undefined = useMemo(() => {
        if (mediaItem === undefined) {
            return undefined;
        }

        return {
            ...mediaItem,
            frame_number: currentFrameIndex,

            // TODO: This logic should be moved to selected media item provider
            type: 'video_frame',
            frame_count: mediaItem.frame_count,
            fps: mediaItem.fps,
            duration: mediaItem.duration,
            // TODO: This should be returned by the backend, atm it's mocked to be 60 fps
            frame_stride: 60,
        };
    }, [currentFrameIndex, mediaItem]);

    const videoControls = useVideoControls(videoRef, playingVideoFrame, changeSelectedMediaItem, setCurrentFrameIndex);

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
        playingVideoFrame !== undefined
            ? {
                  videoFrame: playingVideoFrame,
                  videoRef,
                  videoControls,

                  toggleMute,
                  isMuted,

                  playbackRate,
                  changePlaybackRate,

                  changeCurrentFrameIndex: setCurrentFrameIndex,
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
