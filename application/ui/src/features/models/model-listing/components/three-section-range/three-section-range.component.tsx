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

// Distribute rounded percentages so that they always sum to 100%
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
        { label: 'Training', percentage: trainingPercentage, color: 'var(--moss-tint-1)' },
        { label: 'Validation', percentage: validationPercentage, color: 'var(--brand-daisy-tint)' },
        { label: 'Test', percentage: testingPercentage, color: 'var(--geode-tint)' },
    ];

    return (
        <Flex alignItems={'center'} width={'100%'} data-testid={id}>
            <Text UNSAFE_className={classes.label}>Dataset split</Text>

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

            <Flex gap={'size-150'} alignItems={'center'} UNSAFE_className={classes.label}>
                {labelledPercentages.map(({ label, percentage, color }) => (
                    <Flex key={label} gap={'size-75'} alignItems={'center'}>
                        <span
                            style={{
                                width: 8,
                                height: 8,
                                borderRadius: '50%',
                                backgroundColor: color,
                                display: 'inline-block',
                                flexShrink: 0,
                            }}
                        />
                        <Text UNSAFE_className={classes.label}>{`${label} ${percentage}%`}</Text>
                    </Flex>
                ))}
            </Flex>
        </Flex>
    );
};
