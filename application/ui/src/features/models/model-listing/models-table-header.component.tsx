// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Flex, Grid, Text } from '@geti/ui';
import { SortDown } from '@geti/ui/icons';

import { GRID_COLUMNS } from './constants';
import type { GroupByMode, SortBy } from './types';

interface ModelsTableHeaderProps {
    groupBy: GroupByMode;
    sortBy?: SortBy;
}

const ColumnHeader = ({ label, isSorted }: { label: string; isSorted: boolean }) => (
    <Flex alignItems='center' gap='size-50'>
        <Text>{label}</Text>
        {isSorted && <SortDown width={16} height={16} />}
    </Flex>
);

// NOTE: We cannot have DisclosureGroup inside TableView when using Spectrum so
// We are just rendering the result of the sort, not doing the sort itself on the table.
// The actual sorting comes from the models screen Header.
export const ModelsTableHeader = ({ groupBy, sortBy }: ModelsTableHeaderProps) => {
    return (
        <Grid
            columns={GRID_COLUMNS}
            alignItems={'center'}
            width={'100%'}
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-global-color-gray-200)',
                padding: `${dimensionValue('size-150')} ${dimensionValue('size-600')}
                    ${dimensionValue('size-150')} ${dimensionValue('size-1000')}`,
            }}
        >
            <ColumnHeader label='Model Name' isSorted={sortBy === 'name'} />
            <ColumnHeader label='Trained' isSorted={sortBy === 'trained'} />
            <ColumnHeader
                label={groupBy === 'architecture' ? 'Dataset' : 'Architecture'}
                isSorted={sortBy === 'architecture'}
            />
            <ColumnHeader label='Total size' isSorted={sortBy === 'size'} />
            <ColumnHeader label='Score' isSorted={sortBy === 'score'} />
            <div />
        </Grid>
    );
};
