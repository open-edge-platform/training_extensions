// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Flex } from '@geti/ui';
import { Pause, Play, SoundOff, SoundOn, StepBackward, StepForward } from '@geti/ui/icons';

export const VideoControls = () => {
    const isMuted = false;

    return (
        <Flex alignItems={'center'} gap={'size-100'}>
            <Flex alignItems={'center'}>
                <ActionButton isQuiet aria-label={'Go to previous frame'}>
                    <StepBackward />
                </ActionButton>
                <ActionButton isQuiet aria-label={'Play video'}>
                    <Play />
                </ActionButton>
                <ActionButton isQuiet aria-label={'Pause video'}>
                    <Pause />
                </ActionButton>
                <ActionButton isQuiet aria-label={'Go to next frame'}>
                    <StepForward />
                </ActionButton>
            </Flex>
            <ActionButton isQuiet aria-label={isMuted ? 'Unmute audio' : 'Mute audio'}>
                {isMuted ? <SoundOff /> : <SoundOn />}
            </ActionButton>
        </Flex>
    );
};
