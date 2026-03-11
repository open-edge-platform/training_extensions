// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Flex } from '@geti/ui';
import { Pause, Play, SoundOff, SoundOn, StepBackward, StepForward } from '@geti/ui/icons';

import { useAnnotationActions } from '../../../../shared/annotator/annotation-actions-provider.component';
import { useVideoPlayer } from '../video-player-provider.component';

export const VideoControls = () => {
    const { isMuted, toggleMute, videoControls } = useVideoPlayer();
    const { isPlaying, play, pause, previousFrame, nextFrame, canSelectPreviousFrame, canSelectNextFrame } =
        videoControls;
    const { resetAnnotations } = useAnnotationActions();

    const handlePlay = async () => {
        await play();
        resetAnnotations();
    };

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
                    <ActionButton isQuiet aria-label={'Play video'} onPress={handlePlay}>
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
