// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Grid } from '@geti/ui';

import { ColumnHeader } from '../components/column-header.component';

const RUNNING_JOB_GRID_COLUMNS = ['2fr 1fr minmax(auto, 80px)'];

export const RunningJobTableHeader = () => {
    return (
        <Grid
            columns={RUNNING_JOB_GRID_COLUMNS}
            alignItems={'center'}
            width={'100%'}
            columnGap={'size-200'}
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-global-color-gray-200)',
                padding: `${dimensionValue('size-150')} ${dimensionValue('size-600')}
                    ${dimensionValue('size-150')} ${dimensionValue('size-1000')}`,
            }}
        >
            <ColumnHeader label={'Model Name'} />
            <ColumnHeader label={'Architecture'} />
            <div />
        </Grid>
    );
};
