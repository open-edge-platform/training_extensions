// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, dimensionValue, Flex, Heading, Item, Menu, MenuTrigger, Text } from '@geti/ui';
import { MoreMenu, VideoThumb } from '@geti/ui/icons';

import type { ArchitectureGroup } from '../../types';

type ArchitectureGroupHeaderProps = {
    architecture: ArchitectureGroup;
};

export const ArchitectureGroupHeader = ({ architecture }: ArchitectureGroupHeaderProps) => {
    return (
        <Flex alignItems={'center'} marginBottom={'size-225'} gap={'size-200'}>
            <Heading level={2} UNSAFE_style={{ fontSize: dimensionValue('size-300') }}>
                {architecture.name}
            </Heading>

            <Flex alignItems={'center'} gap={'size-100'}>
                <VideoThumb />
                <Text>{architecture.recommendedFor}</Text>
            </Flex>

            <MenuTrigger>
                <ActionButton isQuiet>
                    <MoreMenu />
                </ActionButton>
                <Menu>
                    <Item key='export'>Export all</Item>
                    <Item key='delete'>Delete all</Item>
                </Menu>
            </MenuTrigger>
        </Flex>
    );
};
