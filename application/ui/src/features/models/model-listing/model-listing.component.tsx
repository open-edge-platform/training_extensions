// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useState } from 'react';

import { Divider, View } from '@geti/ui';

import type { SchemaModelView } from '../../../api/openapi-spec';
import { GroupModelsContainer } from './group-models-container.component';
import { Header } from './header.component';
import type { GroupByMode, SortBy } from './types';
import { groupModelsByArchitecture, groupModelsByDataset } from './utils/grouping';
import { sortModels } from './utils/sorting';

// TODO: Replace with actual API data
const mockModels: SchemaModelView[] = [
    {
        id: 'Amazing model',
        architecture: 'YOLOX',
        parent_revision: null,
        training_info: {
            status: 'successful',
            label_schema_revision: {},
            configuration: {},
        },
        files_deleted: false,
    },
    {
        id: 'Beautiful model',
        architecture: 'ResNet',
        parent_revision: null,
        training_info: {
            status: 'successful',
            label_schema_revision: {},
            configuration: {},
        },
        files_deleted: false,
    },
];

export const ModelListing = () => {
    const [groupBy, setGroupBy] = useState<GroupByMode>('dataset');
    const [sortBy, setSortBy] = useState<SortBy>('score');

    const groupedModels = useMemo(() => {
        const groups = groupBy === 'dataset' ? groupModelsByDataset(mockModels) : groupModelsByArchitecture(mockModels);

        return groups.map((group) => ({ ...group, models: sortModels(group.models, sortBy) }));
    }, [groupBy, sortBy]);

    return (
        <View padding={'size-300'}>
            <Header
                groupBy={groupBy}
                onGroupByChange={setGroupBy}
                onSortChange={(key) => setSortBy(key as SortBy)}
                onPinActiveToggle={() => {}}
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
