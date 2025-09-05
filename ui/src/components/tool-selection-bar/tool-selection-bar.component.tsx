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
                <Selector />
            </IconWrapper>

            <Divider size='S' />

            <IconWrapper isSelected>
                <Polygon />
            </IconWrapper>

            <IconWrapper>
                <SegmentAnythingIcon />
            </IconWrapper>

            <Divider size='S' />

            <IconWrapper>
                <Undo />
            </IconWrapper>

            <IconWrapper>
                <Redo />
            </IconWrapper>
        </>
    );
};

const Settings = () => {
    return (
        <>
            <IconWrapper>
                <Visible />
            </IconWrapper>

            <IconWrapper>
                <Adjustments />
            </IconWrapper>

            <Divider size='S' />

            <IconWrapper>
                <Add />
            </IconWrapper>

            <Flex>110%</Flex>

            <IconWrapper>
                <Remove />
            </IconWrapper>

            <Divider size='S' />

            <IconWrapper>
                <FitScreen />
            </IconWrapper>
        </>
    );
};

export const ToolSelectionBar = () => {
    return (
        <Flex height={'100%'} alignItems={'center'} justifyContent={'center'}>
            <Grid UNSAFE_className={classes.grid}>
                <Flex UNSAFE_className={classes.section}>
                    <Tools />
                </Flex>

                <Flex UNSAFE_className={classes.section}>
                    <Settings />
                </Flex>

                <Flex UNSAFE_className={classes.section}>
                    <IconWrapper>
                        <Hotkeys />
                    </IconWrapper>
                </Flex>
            </Grid>
        </Flex>
    );
};
