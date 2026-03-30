// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, DialogTrigger, Flex, Slider, Text, Tooltip, TooltipTrigger, View } from '@geti-ui/ui';

import { ReactComponent as PlayRate } from '../../../../assets/icons/play-rate.svg';
import { useVideoPlayer } from '../video-player-provider.component';

type PlaybackRate = {
    value: number;
    label: string;
    key: number;
};

const AVAILABLE_PLAYBACK_RATES_MAPPING: Record<number, PlaybackRate> = {
    [1]: {
        value: 0.25,
        label: 'Slower',
        key: 1,
    },
    [2]: {
        value: 0.5,
        label: 'Slow',
        key: 2,
    },
    [3]: {
        value: 1,
        label: 'Normal',
        key: 3,
    },
};

const MIN_RATE = 1;
const MAX_RATE = 3;

export const PlaybackSpeedSlider = () => {
    const { playbackRate, changePlaybackRate } = useVideoPlayer();

    const selectedPlaybackRate =
        Object.values(AVAILABLE_PLAYBACK_RATES_MAPPING).find(({ value }) => value === playbackRate)?.key ?? MAX_RATE;

    return (
        <DialogTrigger type='popover'>
            <TooltipTrigger placement={'bottom'}>
                <ActionButton isQuiet aria-label={'Change playback speed'}>
                    <View padding={'size-100'} width={'size-1000'}>
                        <Flex alignItems={'center'} gap={'size-100'}>
                            <PlayRate />
                            <Text>{AVAILABLE_PLAYBACK_RATES_MAPPING[selectedPlaybackRate].value}x</Text>
                        </Flex>
                    </View>
                </ActionButton>
                <Tooltip>Playback speed</Tooltip>
            </TooltipTrigger>

            <View padding={'size-200'}>
                <Slider
                    id={'playback-speed'}
                    minValue={MIN_RATE}
                    maxValue={MAX_RATE}
                    step={1}
                    label={'Playback speed'}
                    getValueLabel={(value) => AVAILABLE_PLAYBACK_RATES_MAPPING[value].label}
                    aria-label={'Playback'}
                    value={selectedPlaybackRate}
                    onChange={(value) => {
                        changePlaybackRate(AVAILABLE_PLAYBACK_RATES_MAPPING[value].value);
                    }}
                    isFilled
                />
            </View>
        </DialogTrigger>
    );
};
