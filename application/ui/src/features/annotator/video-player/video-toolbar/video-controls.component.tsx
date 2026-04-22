// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Flex } from '@geti/ui';
import { Pause, Play, SoundOff, SoundOn, StepBackward, StepForward } from '@geti/ui/icons';

import { AnnotatorMode } from '../../../../shared/annotator/annotator-mode';
import { useIsFetchingAnyPredictions } from '../../api/use-media-predictions';
import { useVideoPlayer } from '../video-player-provider.component';

type VideoControlsProps = {
    mode: AnnotatorMode;
};

export const VideoControls = ({ mode }: VideoControlsProps) => {
    const { isMuted, toggleMute, videoControls, videoFrame } = useVideoPlayer();
    const { isPlaying, play, pause, previousFrame, nextFrame, canSelectPreviousFrame, canSelectNextFrame } =
        videoControls;

    const isLoadingPredictions = useIsFetchingAnyPredictions(videoFrame.id) && mode === 'prediction';

    return (
        <Flex alignItems={'center'} gap={'size-100'}>
            <Flex alignItems={'center'}>
                <ActionButton
                    isQuiet
                    aria-label={'Go to previous frame'}
                    isDisabled={!canSelectPreviousFrame}
                    onPress={previousFrame}
                >
                    <StepBackward />
                </ActionButton>
                {isPlaying ? (
                    <ActionButton isQuiet aria-label={'Pause video'} onPress={pause}>
                        <Pause />
                    </ActionButton>
                ) : (
                    <ActionButton isQuiet aria-label={'Play video'} onPress={play} isDisabled={isLoadingPredictions}>
                        <Play />
                    </ActionButton>
                )}
                <ActionButton
                    isQuiet
                    aria-label={'Go to next frame'}
                    isDisabled={!canSelectNextFrame}
                    onPress={nextFrame}
                >
                    <StepForward />
                </ActionButton>
            </Flex>
            <ActionButton isQuiet aria-label={isMuted ? 'Unmute audio' : 'Mute audio'} onPress={toggleMute}>
                {isMuted ? <SoundOff /> : <SoundOn />}
            </ActionButton>
        </Flex>
    );
};
