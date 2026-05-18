// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Grid, Item, Key, Picker, Tag, Text } from '@geti/ui';
import { Accept, Search } from '@geti/ui/icons';
import { clsx } from 'clsx';
import { capitalize } from 'lodash-es';

import { DatasetSubset, Media } from '../../../../constants/shared-types';
import { Hotkeys } from '../primary-toolbar/hotkeys/hotkeys.component';
import { Settings } from '../primary-toolbar/settings/settings.component';
import { ToggleFocus } from '../primary-toolbar/toggle-focus.component';
import { ZoomFitScreen } from '../primary-toolbar/zoom/zoom-fit-screen.component';
import { ZoomSelector } from '../primary-toolbar/zoom/zoom-selector.component';
import { Toolbar } from '../toolbar-container/toolbar-container.component';

import classes from './bottom-toolbar.module.scss';

type BottomToolbarProps = {
    mediaItem: Media;
    isUserReviewed?: boolean;
    subset: DatasetSubset;
    onSubsetChange?: (key: Key | null) => void;
    hideHotkeys?: boolean;
    isReadOnlySubset: boolean;
    hasAnnotationStatus?: boolean;
};

export const BottomToolbar = ({
    hideHotkeys,
    mediaItem,
    isUserReviewed,
    subset,
    onSubsetChange,
    isReadOnlySubset,
    hasAnnotationStatus = true,
}: BottomToolbarProps) => {
    const fileName = `${mediaItem.name}.${mediaItem.format} (${mediaItem.width} x ${mediaItem.height} px)`;

    return (
        <Flex justifyContent={'end'}>
            <Toolbar.Container>
                <Grid autoFlow={'column'} autoColumns={'max-content'} gap={'size-50'}>
                    {!hideHotkeys && (
                        <Toolbar.Section>
                            <Hotkeys />
                        </Toolbar.Section>
                    )}

                    <Toolbar.Section>
                        <Flex gap={'size-100'} alignItems={'center'} height={'100%'}>
                            <Text UNSAFE_className={classes.filename}>{fileName}</Text>
                            {hasAnnotationStatus && (
                                <Tag
                                    className={clsx({
                                        [classes.reviewed]: isUserReviewed,
                                        [classes.forReview]: !isUserReviewed,
                                    })}
                                    prefix={isUserReviewed ? <Accept /> : <Search />}
                                    text={isUserReviewed ? 'Reviewed' : 'For Review'}
                                />
                            )}

                            {isReadOnlySubset ? (
                                <Tag withDot={false} text={capitalize(subset)} id={'selected-subset-badge'} />
                            ) : (
                                <Picker
                                    selectedKey={subset}
                                    placeholder={'Select subset'}
                                    aria-label={'Select subset'}
                                    onSelectionChange={onSubsetChange}
                                >
                                    <Item key={'unassigned'}>Unassigned</Item>
                                    <Item key={'validation'}>Validation</Item>
                                    <Item key={'testing'}>Testing</Item>
                                    <Item key={'training'}>Training</Item>
                                </Picker>
                            )}
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
