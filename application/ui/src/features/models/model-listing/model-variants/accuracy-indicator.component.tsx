// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';
import { Pie, PieChart, Sector } from 'recharts';

interface AccuracyIndicatorProps {
    accuracy: number;
}

export const getColor = (accuracy: number): string => {
    if (accuracy >= 75) return 'var(--moss-tint-1)';
    if (accuracy >= 40 && accuracy <= 74) return 'var(--brand-daisy)';

    return 'var(--coral-shade-1)';
};

export const AccuracyIndicator = ({ accuracy }: AccuracyIndicatorProps) => {
    const graphData = [
        { name: 'accuracy', value: accuracy },
        { name: 'remaining', value: 100 - accuracy },
    ];

    return (
        <Flex direction={'column'}>
            <PieChart width={70} height={50}>
                <Pie
                    data={graphData}
                    cx={30}
                    cy={26}
                    startAngle={180}
                    endAngle={0}
                    innerRadius={20}
                    outerRadius={26}
                    dataKey='value'
                    stroke='none'
                    shape={(props) => {
                        const fill = props.index === 0 ? getColor(accuracy) : 'var(--spectrum-global-color-gray-400)';

                        return <Sector {...props} fill={fill} />;
                    }}
                />
                <text
                    x={'50%'}
                    y={'50%'}
                    textAnchor={'middle'}
                    dominantBaseline={'middle'}
                    fill={'var(--spectrum-global-color-gray-900)'}
                    fontSize={'var(--spectrum-global-dimension-size-125)'}
                >
                    {`${accuracy}%`}
                </text>
            </PieChart>
        </Flex>
    );
};
