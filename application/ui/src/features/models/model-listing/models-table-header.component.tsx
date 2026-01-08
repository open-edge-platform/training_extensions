// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Grid, Text } from '@geti/ui';

import { GRID_COLUMNS } from './constants';
import type { GroupByMode } from './types';

interface ModelsTableHeaderProps {
    groupBy: GroupByMode;
}

export const ModelsTableHeader = ({ groupBy }: ModelsTableHeaderProps) => {
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
            <Text>Model Name</Text>
            <Text>Trained</Text>
            <Text>{groupBy === 'architecture' ? 'Dataset' : 'Architecture'}</Text>
            <Text>Total size</Text>
            <Text>Score</Text>
            <div />
        </Grid>
    );
};
