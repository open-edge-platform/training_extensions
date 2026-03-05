// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Grid, Text, View } from '@geti/ui';

import classes from './progress-stepper.module.scss';

export const ProgressStepper = () => {
    return (
        <Grid
            gap={'size-100'}
            rows={['auto', 'auto']}
            alignItems={'center'}
            justifyItems={'center'}
            marginBottom={'size-200'}
            marginX={'size-1000'}
            columns={['30px', '1fr', '30px', '1fr', '30px']}
        >
            <div data-active className={classes.step}>
                1
            </div>
            <View UNSAFE_className={classes.stepDivider}></View>
            <div className={classes.step}>3</div>
            <View UNSAFE_className={classes.stepDivider}></View>
            <div className={classes.step}>5</div>

            <Text>Dataset</Text>
            <div></div>
            <Text UNSAFE_style={{ width: dimensionValue('size-800'), textAlign: 'center' }}>Task type</Text>
            <div></div>
            <Text>Label</Text>
        </Grid>
    );
};
