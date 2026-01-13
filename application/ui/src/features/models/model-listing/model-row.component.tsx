// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, dimensionValue, Flex, Grid, Item, Link, Menu, MenuTrigger, Tag, Text } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';

import type { SchemaModelView } from '../../../api/openapi-spec';
import { ReactComponent as StartIcon } from '../../../assets/icons/start.svg';
import { GRID_COLUMNS } from './constants';
import { AccuracyIndicator } from './model-variants/accuracy-indicator.component';

interface ModelRowProps {
    model: SchemaModelView;
}

export const ModelRow = ({ model }: ModelRowProps) => {
    return (
        <Grid columns={GRID_COLUMNS} alignItems={'center'} width={'100%'}>
            <Flex direction={'column'} gap={'size-50'}>
                <Flex alignItems={'center'} gap={'size-100'}>
                    <Text UNSAFE_style={{ fontSize: dimensionValue('font-size-200') }}>
                        {model.id ?? 'Unnamed Model'}
                    </Text>
                    <Tag
                        prefix={<StartIcon />}
                        style={{
                            backgroundColor: 'var(--energy-blue)',
                            color: 'var(--spectrum-global-color-gray-50)',
                            borderRadius: dimensionValue('size-50'),
                            padding: `${dimensionValue('size-25')} ${dimensionValue('size-50')}`,
                        }}
                        text={'Active'}
                    />
                </Flex>
                <Text
                    UNSAFE_style={{
                        fontSize: dimensionValue('font-size-75'),
                        color: 'var(--spectrum-global-color-gray-700)',
                    }}
                >
                    Fine-tuned from <Link UNSAFE_style={{ textDecoration: 'none' }}>Model Project #1</Link>
                </Text>
            </Flex>

            <Flex direction={'column'} gap={'size-25'}>
                <Text UNSAFE_style={{ fontSize: dimensionValue('font-size-75') }}>01 Oct 2025</Text>
                <Text
                    UNSAFE_style={{
                        fontSize: dimensionValue('font-size-75'),
                        color: 'var(--spectrum-global-color-gray-700)',
                    }}
                >
                    11:07 AM
                </Text>
            </Flex>

            <Text UNSAFE_style={{ fontSize: dimensionValue('font-size-75') }}>{model.architecture}</Text>

            <Text UNSAFE_style={{ fontSize: dimensionValue('font-size-75') }}>500 MB</Text>

            <AccuracyIndicator accuracy={72} />

            <MenuTrigger>
                <ActionButton isQuiet>
                    <MoreMenu />
                </ActionButton>
                <Menu>
                    <Item key='delete'>Delete</Item>
                    <Item key='export'>Export</Item>
                </Menu>
            </MenuTrigger>
        </Grid>
    );
};
