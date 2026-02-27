// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    createContext,
    Dispatch,
    ReactNode,
    RefObject,
    SetStateAction,
    use,
    useLayoutEffect,
    useMemo,
    useRef,
    useState,
} from 'react';

import { VisuallyHidden } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import type { MediaVideoFrame } from '../../../constants/shared-types';
import { getMediaBinaryUrl } from '../../../shared/media-url.utils';
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

    step: number;
    changeStep: Dispatch<SetStateAction<number>>;
};

const useSynchronizeVideoTimeWithInitialFrame = (
    videoRef: RefObject<HTMLVideoElement | null>,
    videoFrame: MediaVideoFrame | undefined
) => {
    /*
     * This part is responsible for updating video time on the initial load when the frame number is not 0.
     * In that case we need to move the video to the correct frame number.
     * */
    useLayoutEffect(() => {
        if (videoFrame?.frame_number == undefined || videoRef.current === null) {
            return;
        }

        if (videoFrame.frame_number > 0 && videoRef.current.currentTime === 0) {
            videoRef.current.currentTime = (videoFrame.frame_number + 1) / videoFrame.fps;
        }
    }, [videoFrame?.frame_number, videoFrame?.fps, videoRef]);
};

const VideoPlayerContext = createContext<VideoPlayerContextProps | null>(null);

type VideoPlayerProviderProps = {
    children: ReactNode;
    videoFrame: MediaVideoFrame | undefined;
    changeSelectedMediaItem: (media: MediaVideoFrame) => void;
};

export const VideoPlayerProvider = ({ children, videoFrame, changeSelectedMediaItem }: VideoPlayerProviderProps) => {
    const projectId = useProjectIdentifier();
    const videoRef = useRef<HTMLVideoElement>(null);
    const [isMuted, setIsMuted] = useState<boolean>(false);
    const [playbackRate, setPlaybackRate] = useState<number>(1);
    const [currentFrameIndex, setCurrentFrameIndex] = useState<number>(videoFrame?.frame_number ?? 0);

    useSynchronizeVideoTimeWithInitialFrame(videoRef, videoFrame);

    const playingVideoFrame: MediaVideoFrame | undefined = useMemo(() => {
        if (videoFrame === undefined) {
            return undefined;
        }

        return {
            ...videoFrame,
            frame_number: currentFrameIndex,
        };
    }, [currentFrameIndex, videoFrame]);

    const [step, setStep] = useState<number>(playingVideoFrame?.frame_stride ?? 1);

    const videoControls = useVideoControls({
        step,
        videoRef,
        videoFrame: playingVideoFrame,
        selectVideoFrame: changeSelectedMediaItem,
        changeCurrentFrameIndex: setCurrentFrameIndex,
    });

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

    const handleEnded = () => {
        if (videoRef.current === null) {
            return;
        }
        videoControls.pause();
        videoRef.current.currentTime = 0;
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

                  step,
                  changeStep: setStep,
              }
            : null;

    return (
        <VideoPlayerContext value={value}>
            {children}
            {playingVideoFrame !== undefined && (
                <VisuallyHidden>
                    <video
                        ref={videoRef}
                        src={getMediaBinaryUrl(projectId, playingVideoFrame.id)}
                        width={playingVideoFrame.width}
                        height={playingVideoFrame.height}
                        preload={'auto'}
                        onEnded={handleEnded}
                        muted
                    />
                </VisuallyHidden>
            )}
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

export const useVideoPlayerContext = () => {
    return use(VideoPlayerContext);
};
