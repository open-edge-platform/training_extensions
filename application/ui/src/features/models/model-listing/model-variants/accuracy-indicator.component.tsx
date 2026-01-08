// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';
import { Cell, Label, Pie, PieChart } from 'recharts';

interface AccuracyIndicatorProps {
    accuracy: number;
}

const getColor = (accuracy: number): string => {
    if (accuracy >= 90) return 'var(--moss-tint-1)';
    if (accuracy >= 70) return 'var(--brand-daisy)';

    return 'var(--coral-shade-1)';
};

export const AccuracyIndicator = ({ accuracy }: AccuracyIndicatorProps) => {
    const graphData = [
        {
            name: 'accuracy',
            value: accuracy,
        },
        {
            name: 'remaining',
            value: 100 - accuracy,
        },
    ];

    return (
        <Flex direction={'column'}>
            <PieChart width={60} height={50}>
                <Pie
                    data={graphData}
                    cx={30}
                    cy={32}
                    startAngle={180}
                    endAngle={0}
                    innerRadius={20}
                    outerRadius={26}
                    dataKey='value'
                    stroke='none'
                >
                    <Cell fill={getColor(accuracy)} />
                    <Cell fill='var(--spectrum-global-color-gray-400)' />
                    <Label
                        value={`${accuracy}%`}
                        position='center'
                        fill='var(--spectrum-global-color-gray-900)'
                        fontSize={10}
                    />
                </Pie>
            </PieChart>
        </Flex>
    );
};
