// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { ActionButton, Divider, Flex, Text, View } from '@geti/ui';
import { ChevronDownLight } from '@geti/ui/icons';
import { clsx } from 'clsx';

import { Toolbar } from '../../../dataset/media-preview/toolbar-container/toolbar-container.component';
import { useVideoPlayer } from '../video-player-provider.component';
import { FrameStep } from './frame-step/frame-step.component';
import { PlaybackSpeedSlider } from './playback-rate.component';
import { VideoAnnotator } from './video-annotator/video-annotator.component';
import { VideoControls } from './video-controls.component';
import { VideoDuration } from './video-duration.component';

import classes from './video-toolbar.module.scss';

export const VideoToolbar = () => {
    const { videoFrame, step, changeStep, videoControls } = useVideoPlayer();
    const [isExpanded, setIsExpanded] = useState(false);

    return (
        <Toolbar.Container>
            <Toolbar.Section>
                <View paddingX={'size-100'}>
                    <Flex alignItems={'center'} justifyContent={'space-between'}>
                        <Flex alignItems={'center'} gap={'size-200'}>
                            <Text>Frames</Text>
                            <VideoControls />
                            <VideoDuration />

                            <Divider orientation={'vertical'} size={'S'} />

                            <FrameStep
                                step={step}
                                onChangeStep={changeStep}
                                isDisabled={videoControls.isPlaying}
                                defaultFps={videoFrame.frame_stride}
                            />

                            <PlaybackSpeedSlider />

                            <Divider orientation={'vertical'} size={'S'} />
                        </Flex>

                        <Flex alignItems={'center'} gap={'size-100'}>
                            <Text>
                                Current frame: {videoFrame.frame_number} / Total frames: {videoFrame.frame_count - 1}
                            </Text>
                            <ActionButton
                                isQuiet
                                onPress={() => setIsExpanded((prev) => !prev)}
                                aria-label={`${isExpanded ? 'Collapse' : 'Expand'} toolbar`}
                            >
                                <ChevronDownLight
                                    className={clsx(classes.chevronButton, {
                                        [classes.chevronButtonCollapsed]: !isExpanded,
                                    })}
                                />
                            </ActionButton>
                        </Flex>
                    </Flex>
                    {isExpanded && <VideoAnnotator />}
                </View>
            </Toolbar.Section>
        </Toolbar.Container>
    );
};
