// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Flex } from '@geti/ui';
import { Pause, Play, SoundOff, SoundOn, StepBackward, StepForward } from '@geti/ui/icons';

import { useVideoPlayer } from '../video-player-provider.component';

export const VideoControls = () => {
    const {
        isMuted,
        isPlaying,
        play,
        pause,
        toggleMute,
        canSelectNextFrame,
        canSelectPreviousFrame,
        nextFrame,
        previousFrame,
    } = useVideoPlayer();

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
                    <ActionButton isQuiet aria-label={'Play video'} onPress={play}>
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
