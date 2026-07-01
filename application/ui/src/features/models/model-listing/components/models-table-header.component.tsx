// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import { dimensionValue, Grid } from '@geti-ui/ui';
import { useProjectTask } from 'hooks/use-project-task.hook';

import { GRID_COLUMNS } from '../constants';
import { useModelListing } from '../provider/model-listing-provider';
import { ColumnHeader } from './column-header.component';
import { getPerformanceColumnLabel } from './model-row/utils';

// NOTE: We cannot have DisclosureGroup inside TableView when using Spectrum so
// We are just rendering the result of the sort, not doing the sort itself on the table.
// The actual sorting comes from the models screen Header.
export const ModelsTableHeader = () => {
    const { groupBy, sortBy, groupedModels } = useModelListing();
    const taskType = useProjectTask();

    const performanceColumnName = useMemo(() => {
        const models = groupedModels.flatMap((group) => group.models);

        return getPerformanceColumnLabel(models, taskType);
    }, [groupedModels, taskType]);

    return (
        <Grid
            columns={GRID_COLUMNS}
            alignItems={'center'}
            width={'100%'}
            columnGap={'size-200'}
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
                isSorted={sortBy === 'architecture' || sortBy === 'dataset'}
            />
            <ColumnHeader label='Total size' isSorted={sortBy === 'size'} />
            <ColumnHeader label={performanceColumnName} isSorted={sortBy === 'score'} />
            <div />
        </Grid>
    );
};
