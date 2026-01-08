// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Grid, Item, Picker, Text, ToggleButton } from '@geti/ui';
import { Search } from '@geti/ui/icons';

import type { GroupByMode } from './types';

interface HeaderProps {
    groupBy?: GroupByMode;
    onGroupByChange: (groupBy: GroupByMode) => void;
    onSortChange: (sortBy: string) => void;
    onPinActiveToggle: (isPinned: boolean) => void;
}

export const Header = ({ groupBy, onGroupByChange, onSortChange, onPinActiveToggle }: HeaderProps) => {
    return (
        <Grid columns={['auto auto 1fr auto']} gap={'size-100'} alignItems={'center'}>
            <Text>Models</Text>

            <Flex marginStart={'size-300'} gap={'size-100'}>
                <Picker
                    placeholder={'Group by'}
                    width={'size-3000'}
                    aria-label={'Group models'}
                    selectedKey={groupBy}
                    onSelectionChange={(key) => onGroupByChange(key as GroupByMode)}
                >
                    <Item key='dataset'>Group by: Dataset</Item>
                    <Item key='architecture'>Group by: Architecture</Item>
                </Picker>
                <Picker
                    placeholder={'Sort by'}
                    width={'size-3000'}
                    aria-label={'Sort models'}
                    defaultSelectedKey={'active-model'}
                    onSelectionChange={(key) => onSortChange(key as string)}
                >
                    <Item key='active-model'>Sort: Active model</Item>
                    <Item key='architecture'>Sort: Architecture</Item>
                </Picker>
            </Flex>

            <Flex>
                <ToggleButton isEmphasized onChange={onPinActiveToggle}>
                    Pin active model on top
                </ToggleButton>
            </Flex>

            <Flex>
                <Search />
            </Flex>
        </Grid>
    );
};
