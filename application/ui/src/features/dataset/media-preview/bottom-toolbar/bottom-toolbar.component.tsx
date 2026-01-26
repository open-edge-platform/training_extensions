// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Grid, Item, Picker, Tag, Text } from '@geti/ui';
import { Search } from '@geti/ui/icons';

import { Hotkeys } from '../primary-toolbar/hotkeys/hotkeys.component';
import { Settings } from '../primary-toolbar/settings/settings.component';
import { ToggleFocus } from '../primary-toolbar/toggle-focus.component';
import { ZoomFitScreen } from '../primary-toolbar/zoom/zoom-fit-screen.component';
import { ZoomSelector } from '../primary-toolbar/zoom/zoom-selector.component';

import classes from '../media-preview.module.scss';

export const BottomToolbar = () => {
    return (
        <Flex justifyContent={'end'}>
            <Grid UNSAFE_className={classes.toolbarGrid} autoFlow={'column'} autoColumns={'max-content'}>
                <Flex UNSAFE_className={classes.toolbarSection}>
                    <Hotkeys />
                </Flex>

                <Flex UNSAFE_className={classes.toolbarSection} gap={'size-100'}>
                    <Text UNSAFE_className={classes.filename}>VID_20210209_160431.jpg (1080x1920 px)</Text>
                    <Tag style={{ backgroundColor: 'var(--coral-shade-1)' }} prefix={<Search />} text={'For Review'} />
                    <Picker
                        // selectedKey={selectedLabel?.id}
                        placeholder={'Select subset'}
                        // onSelectionChange={onSelect}
                        aria-label='Subset picker'
                    >
                        <Item>Validation</Item>
                        <Item>Testing</Item>
                        <Item>Training</Item>
                    </Picker>
                </Flex>

                <Flex UNSAFE_className={classes.toolbarSection}>
                    <Settings />

                    <ZoomSelector />

                    <ToggleFocus />

                    <ZoomFitScreen />
                </Flex>
            </Grid>
        </Flex>
    );
};
