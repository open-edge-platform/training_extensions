// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Divider, Flex, View } from '@geti/ui';

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
    const [expandedModelIds, setExpandedModelIds] = useState<Set<string>>(new Set());

    const handleExpandModel = (modelId: string) => {
        setExpandedModelIds((prev) => {
            const newExpandedModelIds = new Set(prev);

            if (newExpandedModelIds.has(modelId)) {
                newExpandedModelIds.delete(modelId);
            } else {
                newExpandedModelIds.add(modelId);
            }

            return newExpandedModelIds;
        });
    };

    return (
        <View padding={'size-300'}>
            <Header
                groupBy={groupBy}
                onGroupByChange={onGroupByChange}
                onSortChange={onSortChange}
                onPinActiveToggle={onPinActiveToggle}
            />

            <Divider size={'S'} marginY={'size-300'} />

            <Flex direction={'column'} gap={'size-300'}>
                {groupedModels.map(({ group, models }, index) => (
                    <GroupModelsContainer
                        key={'id' in group ? group.id : `${group.name}-${index}`}
                        groupBy={groupBy}
                        group={group}
                        models={models}
                        sortBy={sortBy}
                        expandedModelIds={expandedModelIds}
                        onExpandModel={handleExpandModel}
                    />
                ))}
            </Flex>
        </View>
    );
};
