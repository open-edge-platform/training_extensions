// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Text } from '@geti/ui';
import { useNumberFormatter } from 'react-aria';

type ChangeType = 'accuracy' | 'size';

const getDeltaColor = (change: number, changeType: ChangeType): string => {
    const isPositiveOutcome = (changeType === 'accuracy' && change > 0) || (changeType === 'size' && change < 0);

    return isPositiveOutcome ? 'var(--moss-tint-1)' : 'var(--coral-shade-1)';
};

type ModelVariantDeltaProps = {
    currentValue: number | undefined;
    baselineValue: number | undefined;
    changeType?: ChangeType;
};
export const ModelVariantDelta = ({ currentValue, baselineValue, changeType = 'accuracy' }: ModelVariantDeltaProps) => {
    const formatter = useNumberFormatter({
        style: 'percent',
        maximumFractionDigits: 0,
        signDisplay: 'exceptZero',
    });

    if (
        currentValue === undefined ||
        baselineValue === undefined ||
        baselineValue === 0 ||
        currentValue === baselineValue
    ) {
        return null;
    }

    // Relative percentage delta: (current - baseline) / baseline
    const delta = (currentValue - baselineValue) / baselineValue;
    const testId = `model-variant-delta-${changeType}`;

    return (
        <Text data-testid={testId} UNSAFE_style={{ color: getDeltaColor(delta, changeType) }}>
            {formatter.format(delta)}
        </Text>
    );
};

type ValueWithDeltaProps = {
    value: number | undefined;
    baselineValue: number | undefined;
    changeType?: 'accuracy' | 'size';
    displayValue: string;
    showDelta: boolean;
    precision?: string;
};

export const ValueWithDelta = ({
    value,
    baselineValue,
    changeType = 'accuracy',
    displayValue,
    showDelta,
    precision,
}: ValueWithDeltaProps) => {
    const testId = precision ? `model-variant-value-${changeType}-${precision}` : `model-variant-value-${changeType}`;

    return (
        <Flex direction={'column'} gap={'size-25'}>
            <Text data-testid={testId}>{displayValue}</Text>
            {showDelta && (
                <ModelVariantDelta currentValue={value} baselineValue={baselineValue} changeType={changeType} />
            )}
        </Flex>
    );
};
