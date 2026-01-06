// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Button, Flex, Grid, Heading, Item, Menu, MenuTrigger, Text } from '@geti/ui';
import { Image, MoreMenu, Tag } from '@geti/ui/icons';

import { ThreeSectionRange } from './three-section-range.component';

import classes from './dataset-item.module.scss';

export const DatasetItem = () => {
    return (
        <Grid
            columns={['auto', '1fr', 'auto', '1fr', 'auto']}
            alignItems={'center'}
            marginBottom={'size-225'}
            gap={'size-200'}
        >
            <Flex alignItems={'center'} gap={'size-50'}>
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
                        <Item key='export'>Export</Item>
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

            {/* TODO: Update with actual values from the API  */}
            <ThreeSectionRange trainingValue={7} validationValue={2} testingValue={1} />

            <Flex>
                <Button variant='primary'>Train model</Button>
            </Flex>
        </Grid>
    );
};
