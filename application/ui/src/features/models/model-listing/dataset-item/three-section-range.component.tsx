// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Grid, Text, View } from '@geti/ui';

import classes from './dataset-item.module.scss';

type ThreeSectionRangeProps = {
    trainingValue: number;
    validationValue: number;
    testingValue: number;
};

export const ThreeSectionRange = ({ trainingValue, validationValue, testingValue }: ThreeSectionRangeProps) => {
    const gridColumns = [
        trainingValue > 0 ? `${trainingValue}fr` : '1fr',
        validationValue > 0 ? `${validationValue}fr` : '1fr',
        testingValue > 0 ? `${testingValue}fr` : '1fr',
    ];

    return (
        <Flex alignItems={'center'} width={'100%'}>
            <Text UNSAFE_className={classes.label}>TRAINING SUBSETS</Text>

            <Grid
                columns={gridColumns}
                width='100%'
                height={'size-100'}
                marginStart={'size-200'}
                marginEnd={'size-50'}
                UNSAFE_className={classes.rangeGrid}
            >
                {trainingValue > 0 && <View height='100%' UNSAFE_style={{ backgroundColor: 'var(--moss-tint-1)' }} />}
                {validationValue > 0 && (
                    <View height='100%' UNSAFE_style={{ backgroundColor: 'var(--brand-daisy-tint)' }} />
                )}
                {testingValue > 0 && <View height='100%' UNSAFE_style={{ backgroundColor: 'var(--geode-tint)' }} />}
            </Grid>

            <Text UNSAFE_className={classes.label}>{`${trainingValue}% / ${validationValue}% / ${testingValue}%`}</Text>
        </Flex>
    );
};
