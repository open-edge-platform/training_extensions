// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Grid, Item, Picker, Tag, Text } from '@geti/ui';
import { Search } from '@geti/ui/icons';

import { Hotkeys } from '../primary-toolbar/hotkeys/hotkeys.component';
import { Settings } from '../primary-toolbar/settings/settings.component';
import { ToggleFocus } from '../primary-toolbar/toggle-focus.component';
import { ZoomFitScreen } from '../primary-toolbar/zoom/zoom-fit-screen.component';
import { ZoomSelector } from '../primary-toolbar/zoom/zoom-selector.component';
import { Toolbar } from '../toolbar-container/toolbar-container.component';

import styles from './bottom-toolbar.module.scss';

export const BottomToolbar = () => {
    return (
        <Flex justifyContent={'end'}>
            <Toolbar.Container>
                <Grid autoFlow={'column'} autoColumns={'max-content'} gap={'size-50'}>
                    <Toolbar.Section>
                        <Hotkeys />
                    </Toolbar.Section>

                    <Toolbar.Section>
                        <Flex gap={'size-100'} alignItems={'center'}>
                            <Text UNSAFE_className={styles.filename}>VID_20210209_160431.jpg (1080x1920 px)</Text>
                            <Tag className={styles.forReview} prefix={<Search />} text={'For Review'} />
                            <Picker placeholder={'Select subset'} aria-label='Subset picker'>
                                <Item>Validation</Item>
                                <Item>Testing</Item>
                                <Item>Training</Item>
                            </Picker>
                        </Flex>
                    </Toolbar.Section>

                    <Toolbar.Section>
                        <Flex alignItems={'center'}>
                            <Settings />

                            <ZoomSelector />

                            <ToggleFocus />

                            <ZoomFitScreen />
                        </Flex>
                    </Toolbar.Section>
                </Grid>
            </Toolbar.Container>
        </Flex>
    );
};
