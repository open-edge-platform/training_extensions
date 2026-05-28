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

    const labelledPercentages = [
        { label: 'Training', percentage: trainingPercentage, color: 'var(--moss-tint-1)' },
        { label: 'Validation', percentage: validationPercentage, color: 'var(--brand-daisy-tint)' },
        { label: 'Test', percentage: testingPercentage, color: 'var(--geode-tint)' },
    ];

    return (
        <Flex alignItems={'center'} data-testid={id}>
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
