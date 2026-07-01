// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, View } from '@geti/ui';
import { CartesianGrid, Line, LineChart, Tooltip, XAxis, YAxis } from 'recharts';

import { Box } from '../components/box/box.component';

export type MetricGraphPoint = {
    x: number;
    y: number;
};

type MetricGraphProps = {
    title: string;
    data?: MetricGraphPoint[];
    xAxisLabel?: string;
    yAxisLabel: string;
};

const X_AXIS_TICK_COUNT = 8;
const Y_AXIS_TICK_COUNT = 4;

export const MetricGraph = ({ title, data, xAxisLabel, yAxisLabel }: MetricGraphProps) => {
    return (
        <Flex flex={1} direction={'column'} minWidth={'size-5000'}>
            <Box
                title={title}
                content={
                    <View backgroundColor={'gray-50'} minHeight={'size-3800'}>
                        <LineChart
                            responsive
                            width={'100%'}
                            style={{ aspectRatio: 1.6 }}
                            data={data}
                            margin={{ top: 35, bottom: 35, left: 35 }}
                        >
                            <CartesianGrid />
                            <XAxis
                                dataKey='x'
                                type='number'
                                domain={[0, 'auto']}
                                label={{ value: xAxisLabel ?? 'x', position: 'bottom', fill: '#666', offset: 12 }}
                                tickCount={X_AXIS_TICK_COUNT}
                                tickMargin={12}
                            />
                            <YAxis
                                label={{ value: yAxisLabel, angle: -90, position: 'center', dx: -38, fill: '#666' }}
                                tickCount={Y_AXIS_TICK_COUNT}
                                tickMargin={12}
                                tickFormatter={(value) => Number(value).toFixed(4)}
                            />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#fff', border: '1px solid #ccc' }}
                                labelStyle={{ color: '#333' }}
                            />
                            <Line
                                type='linear'
                                dataKey='y'
                                name={yAxisLabel}
                                stroke='var(--energy-blue)'
                                strokeWidth={2}
                                dot={false}
                            />
                        </LineChart>
                    </View>
                }
            />
        </Flex>
    );
};
