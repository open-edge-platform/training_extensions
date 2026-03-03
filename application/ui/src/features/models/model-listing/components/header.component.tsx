// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Grid, Item, Picker, ToggleButton } from '@geti/ui';

import { TrainModel } from '../../train-model/train-model.component';
import { useModelListing } from '../provider/model-listing-provider';
import type { GroupByMode, SortBy } from '../types';
import { ExpandableSearch } from './expandable-search/expandable-search.component';

export const Header = () => {
    const {
        groupBy,
        sortBy,
        onGroupByChange,
        onSortChange,
        onPinActiveToggle,
        searchBy,
        onSearchChange,
        pinActive,
        showFailedModels,
        onToggleShowFailedModels,
    } = useModelListing();

    return (
        <Grid columns={['auto auto 1fr auto']} gap={'size-100'} alignItems={'center'}>
            <Flex gap={'size-100'}>
                <Picker
                    placeholder={'Group by'}
                    width={'size-2400'}
                    aria-label={'Group models'}
                    selectedKey={groupBy}
                    onSelectionChange={(key) => onGroupByChange(key as GroupByMode)}
                >
                    <Item key='dataset'>Group by: Dataset</Item>
                    <Item key='architecture'>Group by: Architecture</Item>
                </Picker>
                <Picker
                    placeholder={'Sort by'}
                    width={'size-2000'}
                    aria-label={'Sort models'}
                    selectedKey={sortBy}
                    onSelectionChange={(key) => onSortChange(key as SortBy)}
                >
                    <Item key='name'>Sort: Name</Item>
                    <Item key='trained'>Sort: Trained</Item>
                    <Item key='architecture'>Sort: Architecture</Item>
                    <Item key='size'>Sort: Size</Item>
                    <Item key='score'>Sort: Score</Item>
                </Picker>
            </Flex>

            <Flex gap={'size-100'}>
                <ToggleButton isEmphasized isSelected={pinActive} onChange={onPinActiveToggle}>
                    Pin active model on top
                </ToggleButton>

                <ToggleButton isEmphasized isSelected={showFailedModels} onChange={onToggleShowFailedModels}>
                    Show failed models
                </ToggleButton>
            </Flex>

            <Flex marginStart={'auto'} gap={'size-100'}>
                <ExpandableSearch value={searchBy} onChange={onSearchChange} />
                <TrainModel />
            </Flex>
        </Grid>
    );
};
