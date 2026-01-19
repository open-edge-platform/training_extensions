// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Grid, Item, Picker, Text, ToggleButton } from '@geti/ui';

import { useModelListing } from '../provider/model-listing-provider';
import type { GroupByMode, SortBy } from '../types';
import { ExpandableSearch } from './expandable-search/expandable-search.component';

export const Header = () => {
    const { groupBy, onGroupByChange, onSortChange, onPinActiveToggle, searchBy, onSearchChange } = useModelListing();

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
                    defaultSelectedKey={'trained'}
                    onSelectionChange={(key) => onSortChange(key as SortBy)}
                >
                    <Item key='name'>Sort: Name</Item>
                    <Item key='trained'>Sort: Trained</Item>
                    <Item key='architecture'>Sort: Architecture</Item>
                    <Item key='size'>Sort: Size</Item>
                    <Item key='score'>Sort: Score</Item>
                </Picker>
            </Flex>

            <Flex>
                <ToggleButton isEmphasized onChange={onPinActiveToggle}>
                    Pin active model on top
                </ToggleButton>
            </Flex>

            <Flex>
                <ExpandableSearch value={searchBy} onChange={onSearchChange} />
            </Flex>
        </Grid>
    );
};
