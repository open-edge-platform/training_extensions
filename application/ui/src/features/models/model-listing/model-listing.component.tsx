// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useState } from 'react';

import { Divider, View } from '@geti/ui';

import type { SchemaModelView } from '../../../api/openapi-spec';
import { GroupModelsContainer } from './group-models-container.component';
import { Header } from './header.component';
import type { GroupByMode, GroupedModels } from './types';

// TODO: Replace with actual API data
const mockModels: SchemaModelView[] = [
    {
        id: 'model-1',
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
        id: 'model-2',
        architecture: 'YOLOX',
        parent_revision: null,
        training_info: {
            status: 'successful',
            label_schema_revision: {},
            configuration: {},
        },
        files_deleted: false,
    },
];
const mockGroupedModels: GroupedModels[] = [
    {
        group: {
            id: 'dataset-1',
            name: 'Dataset #dataset-1',
            createdAt: 'Created 01 Oct 2025, 11:07 AM',
            labelCount: 2,
            imageCount: 3600,
            trainingSubsets: {
                training: 70,
                validation: 20,
                testing: 10,
            },
        },
        models: mockModels,
    },
];

// TODO: implement grouping
const groupModelsByDataset = (): GroupedModels[] => {
    return mockGroupedModels;
};

const groupModelsByArchitecture = (): GroupedModels[] => {
    return mockGroupedModels;
};

export const ModelListing = () => {
    const [groupBy, setGroupBy] = useState<GroupByMode>('dataset');

    const groupedModels = useMemo(() => {
        if (groupBy === 'dataset') {
            return groupModelsByDataset();
        }

        return groupModelsByArchitecture();
    }, [groupBy]);

    return (
        <View padding={'size-300'}>
            <Header
                groupBy={groupBy}
                onGroupByChange={setGroupBy}
                onSortChange={() => {}}
                onPinActiveToggle={() => {}}
            />

            <Divider size={'S'} marginY={'size-300'} />

            {groupedModels.map(({ group, models }, index) => (
                <GroupModelsContainer
                    key={'id' in group ? group.id : `${group.name}-${index}`}
                    groupBy={groupBy}
                    group={group}
                    models={models}
                />
            ))}
        </View>
    );
};
