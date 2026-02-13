// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { ActionButton, Flex, Text, View } from '@geti/ui';
import { ChevronDownLight } from '@geti/ui/icons';
import { clsx } from 'clsx';

import { Toolbar } from '../../../dataset/media-preview/toolbar-container/toolbar-container.component';
import { VideoControls } from './video-controls.component';
import { VideoDuration } from './video-duration.component';

import classes from './video-toolbar.module.scss';

export const VideoToolbar = () => {
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
                        </Flex>

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
                </View>
            </Toolbar.Section>
        </Toolbar.Container>
    );
};
