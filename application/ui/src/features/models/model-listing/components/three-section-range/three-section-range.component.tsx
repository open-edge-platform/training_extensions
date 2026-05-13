// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Grid, Text } from '@geti/ui';

import classes from './three-section-range.module.scss';

type ThreeSectionRangeProps = {
    id?: string;
    trainingValue: number;
    validationValue: number;
    testingValue: number;
};

// Distribute rounded percentages so that they always sum to 100% (largest
// remainder method). Independent rounding of each percentage can otherwise
// produce totals like 101% (e.g. 65.4 / 23.6 / 11.7 -> 65 / 24 / 12).
const computeRoundedPercentages = (values: number[]): number[] => {
    const total = values.reduce((sum, value) => sum + value, 0);

    if (total <= 0) {
        return values.map(() => 0);
    }

    const exactPercentages = values.map((value) => (value * 100) / total);
    const flooredPercentages = exactPercentages.map((percentage) => Math.floor(percentage));
    let remainder = 100 - flooredPercentages.reduce((sum, value) => sum + value, 0);

    const indicesByRemainder = exactPercentages
        .map((percentage, index) => ({ index, fractional: percentage - Math.floor(percentage) }))
        .sort((a, b) => b.fractional - a.fractional);

    const result = [...flooredPercentages];
    for (const { index } of indicesByRemainder) {
        if (remainder <= 0) break;
        result[index] += 1;
        remainder -= 1;
    }

    return result;
};

export const ThreeSectionRange = ({ id, trainingValue, validationValue, testingValue }: ThreeSectionRangeProps) => {
    const [trainingPercentage, validationPercentage, testingPercentage] = computeRoundedPercentages([
        trainingValue,
        validationValue,
        testingValue,
    ]);

    const gridColumns = [
        trainingValue > 0 ? `${trainingValue}fr` : '1fr',
        validationValue > 0 ? `${validationValue}fr` : '1fr',
        testingValue > 0 ? `${testingValue}fr` : '1fr',
    ];

    const labelledPercentages = [
        { label: 'Training', percentage: trainingPercentage },
        { label: 'Validation', percentage: validationPercentage },
        { label: 'Test', percentage: testingPercentage },
    ];

    return (
        <Flex alignItems={'center'} width={'100%'} data-testid={id}>
            <Text UNSAFE_className={classes.label}>TRAINING SUBSETS</Text>

            <Grid
                columns={gridColumns}
                width='100%'
                height={'size-100'}
                marginStart={'size-200'}
                marginEnd={'size-50'}
                UNSAFE_className={classes.rangeGrid}
            >
                {trainingValue > 0 && (
                    <div
                        style={{ height: '100%', backgroundColor: 'var(--moss-tint-1)' }}
                        aria-label={`Training: ${trainingPercentage}%`}
                    />
                )}
                {validationValue > 0 && (
                    <div
                        style={{ height: '100%', backgroundColor: 'var(--brand-daisy-tint)' }}
                        aria-label={`Validation: ${validationPercentage}%`}
                    />
                )}
                {testingValue > 0 && (
                    <div
                        style={{ height: '100%', backgroundColor: 'var(--geode-tint)' }}
                        aria-label={`Test: ${testingPercentage}%`}
                    />
                )}
            </Grid>

            <Text UNSAFE_className={classes.label}>
                {labelledPercentages.map(({ label, percentage }) => `${label} ${percentage}%`).join(' / ')}
            </Text>
        </Flex>
    );
};
