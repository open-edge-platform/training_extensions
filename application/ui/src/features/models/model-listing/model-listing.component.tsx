// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider, View } from '@geti/ui';

import { GroupModelsContainer } from './components/group-models-container.component';
import { Header } from './components/header.component';
import type { GroupByMode, GroupedModels, SortBy } from './types';

type ModelListingProps = {
    groupedModels: GroupedModels[];
    groupBy: GroupByMode;
    sortBy: SortBy;
    onGroupByChange: (mode: GroupByMode) => void;
    onSortChange: (key: SortBy) => void;
    onPinActiveToggle: () => void;
};
export const ModelListing = ({
    groupedModels,
    groupBy,
    sortBy,
    onGroupByChange,
    onSortChange,
    onPinActiveToggle,
}: ModelListingProps) => {
    return (
        <View padding={'size-300'}>
            <Header
                groupBy={groupBy}
                onGroupByChange={onGroupByChange}
                onSortChange={onSortChange}
                onPinActiveToggle={onPinActiveToggle}
            />

            <Divider size={'S'} marginY={'size-300'} />

            {groupedModels.map(({ group, models }, index) => (
                <GroupModelsContainer
                    key={'id' in group ? group.id : `${group.name}-${index}`}
                    groupBy={groupBy}
                    group={group}
                    models={models}
                    sortBy={sortBy}
                />
            ))}
        </View>
    );
};
