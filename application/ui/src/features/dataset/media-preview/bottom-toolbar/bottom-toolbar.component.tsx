// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Grid, Item, Picker, Tag, Text } from '@geti/ui';
import { Accept, Search } from '@geti/ui/icons';
import { clsx } from 'clsx';

import { Media } from '../../../../constants/shared-types';
import { Hotkeys } from '../primary-toolbar/hotkeys/hotkeys.component';
import { Settings } from '../primary-toolbar/settings/settings.component';
import { ToggleFocus } from '../primary-toolbar/toggle-focus.component';
import { ZoomFitScreen } from '../primary-toolbar/zoom/zoom-fit-screen.component';
import { ZoomSelector } from '../primary-toolbar/zoom/zoom-selector.component';
import { Toolbar } from '../toolbar-container/toolbar-container.component';

import classes from './bottom-toolbar.module.scss';

type BottomToolbarProps = {
    isUserReviewed: boolean;
    mediaItem: Media;
};

export const BottomToolbar = ({ isUserReviewed, mediaItem }: BottomToolbarProps) => {
    const fileName = `${mediaItem.name}.${mediaItem.format} (${mediaItem.width} x ${mediaItem.height} px)`;

    return (
        <Flex justifyContent={'end'}>
            <Toolbar.Container>
                <Grid autoFlow={'column'} autoColumns={'max-content'} gap={'size-50'}>
                    <Toolbar.Section>
                        <Hotkeys />
                    </Toolbar.Section>

                    <Toolbar.Section>
                        <Flex gap={'size-100'} alignItems={'center'}>
                            <Text UNSAFE_className={classes.filename}>{fileName}</Text>
                            <Tag
                                className={clsx({
                                    [classes.accepted]: isUserReviewed,
                                    [classes.forReview]: !isUserReviewed,
                                })}
                                prefix={isUserReviewed ? <Accept /> : <Search />}
                                text={isUserReviewed ? 'Accepted' : 'For Review'}
                            />

                            {/* TODO: Update these once backend is ready */}
                            <Picker placeholder={'Select subset'} aria-label={'Select subset'}>
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
