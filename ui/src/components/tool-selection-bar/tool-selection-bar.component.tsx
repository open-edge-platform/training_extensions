// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { Divider, Flex, Grid } from '@geti/ui';
import {
    Add,
    Adjustments,
    FitScreen,
    Hotkeys,
    Polygon,
    Redo,
    Remove,
    SegmentAnythingIcon,
    Selector,
    Undo,
    Visible,
} from '@geti/ui/icons';

import classes from './tool-selection-bar.module.scss';

const ICON_DIMENSIONS = {
    width: 16,
    height: 16,
};

const IconWrapper = ({ children, isSelected }: { children: ReactNode; isSelected?: boolean }) => {
    return (
        <Flex
            UNSAFE_className={classes.iconWrapper}
            UNSAFE_style={{
                backgroundColor: isSelected ? 'var(--energy-blue)' : 'transparent',
                fill: isSelected ? 'var(--spectrum-global-color-gray-50)' : 'white',
            }}
        >
            {children}
        </Flex>
    );
};

const Tools = () => {
    return (
        <>
            <IconWrapper>
                <Selector {...ICON_DIMENSIONS} />
            </IconWrapper>

            <Divider size='S' />

            <IconWrapper isSelected>
                <Polygon {...ICON_DIMENSIONS} />
            </IconWrapper>

            <IconWrapper>
                <SegmentAnythingIcon {...ICON_DIMENSIONS} />
            </IconWrapper>

            <Divider size='S' />

            <IconWrapper>
                <Undo {...ICON_DIMENSIONS} />
            </IconWrapper>

            <IconWrapper>
                <Redo {...ICON_DIMENSIONS} />
            </IconWrapper>
        </>
    );
};

const Settings = () => {
    return (
        <>
            <IconWrapper>
                <Visible {...ICON_DIMENSIONS} />
            </IconWrapper>

            <IconWrapper>
                <Adjustments {...ICON_DIMENSIONS} />
            </IconWrapper>

            <Divider size='S' />

            <IconWrapper>
                <Add {...ICON_DIMENSIONS} />
            </IconWrapper>

            <Flex>110%</Flex>

            <IconWrapper>
                <Remove {...ICON_DIMENSIONS} />
            </IconWrapper>

            <Divider size='S' />

            <IconWrapper>
                <FitScreen {...ICON_DIMENSIONS} />
            </IconWrapper>
        </>
    );
};

export const ToolSelectionBar = () => {
    return (
        <Flex height={'100%'} alignItems={'center'} justifyContent={'center'}>
            <Grid
                rows={'auto auto auto'}
                width={'size-600'}
                gap={'size-50'}
                alignItems={'center'}
                UNSAFE_style={{
                    backgroundColor: 'var(--spectrum-global-color-gray-200)',
                    borderRadius: 'var(--spectrum-global-dimension-size-100)',
                    padding: 'var(--spectrum-global-dimension-size-50)',
                }}
            >
                <Flex UNSAFE_className={classes.section}>
                    <Tools />
                </Flex>

                <Flex UNSAFE_className={classes.section}>
                    <Settings />
                </Flex>

                <Flex UNSAFE_className={classes.section}>
                    <IconWrapper>
                        <Hotkeys {...ICON_DIMENSIONS} />
                    </IconWrapper>
                </Flex>
            </Grid>
        </Flex>
    );
};
