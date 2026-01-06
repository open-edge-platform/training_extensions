// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Flex, Grid, Heading, Item, Menu, MenuTrigger, RangeSlider, Text } from '@geti/ui';
import { Image, MoreMenu, Tag } from '@geti/ui/icons';

import classes from './models.module.scss';

export const DatasetHeader = () => {
    return (
        <Grid
            columns={['auto', 'auto', 'minmax(0, 1fr)', 'auto']}
            alignItems={'center'}
            marginBottom={'size-225'}
            gap={'size-200'}
        >
            <Flex alignItems={'center'}>
                <Heading level={2} UNSAFE_style={{ fontSize: 'var(--spectrum-global-dimension-size-300)' }}>
                    Dataset #1
                </Heading>
                <MenuTrigger onOpenChange={() => {}}>
                    <ActionButton isQuiet>
                        <MoreMenu />
                    </ActionButton>
                    <Menu>
                        <Item key='rename'>Rename</Item>
                        <Item key='delete'>Delete</Item>
                    </Menu>
                </MenuTrigger>
            </Flex>
            <Text
                UNSAFE_style={{
                    color: 'var(--spectrum-global-color-gray-700)',
                }}
            >
                Created 01 Oct 2025, 11:07 AM
            </Text>

            <Flex gap={'size-50'} justifyContent={'center'}>
                <Flex UNSAFE_className={classes.tag}>
                    <Tag /> 2
                </Flex>
                <Flex UNSAFE_className={classes.tag}>
                    <Image /> 3,600
                </Flex>
            </Flex>

            <Flex>
                <RangeSlider labelPosition={'side'} label={'TRAINING SUBSETS'} showValueLabel />
            </Flex>
        </Grid>
    );
};
