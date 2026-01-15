// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Flex, Heading, Item, Menu, MenuTrigger, View } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

type MetricGraphProps<T extends Record<string, unknown>> = {
    title: string;
    data?: T[];
    dataKey: keyof T & string;
    xAxisKey?: keyof T & string;
    yAxisLabel: string;
    yAxisDomain?: [number, number];
    yAxisTicks?: number[];
    xAxisTicks?: number[];
};

const DEFAULT_TICKS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 43];
export const MetricGraph = <T extends Record<string, unknown>>({
    title,
    data,
    dataKey,
    xAxisKey = 'epoch' as keyof T & string,
    yAxisLabel,
    yAxisDomain,
    yAxisTicks,
    xAxisTicks = DEFAULT_TICKS,
}: MetricGraphProps<T>) => {
    return (
        <Flex flex={1} direction={'column'}>
            <Flex
                justifyContent={'space-between'}
                alignItems={'center'}
                UNSAFE_style={{
                    backgroundColor: 'var(--spectrum-global-color-gray-200)',
                    padding: 'var(--spectrum-global-dimension-size-50) var(--spectrum-global-dimension-size-200)',
                }}
            >
                <Heading level={5}>{title}</Heading>
                <MenuTrigger>
                    <ActionButton isQuiet>
                        <MoreMenu />
                    </ActionButton>
                    <Menu>
                        <Item key='delete'>Delete</Item>
                        <Item key='export'>Export</Item>
                    </Menu>
                </MenuTrigger>
            </Flex>
            <View
                flex={1}
                paddingY={'size-200'}
                paddingX={'size-550'}
                backgroundColor={'gray-50'}
                minHeight={'size-3400'}
            >
                <ResponsiveContainer width='100%' height='100%'>
                    <AreaChart data={data} margin={{ top: 35, bottom: 35, left: 20 }}>
                        <CartesianGrid />
                        <XAxis
                            dataKey={xAxisKey}
                            label={{ value: xAxisKey, position: 'bottom', fill: '#666', offset: 12 }}
                            ticks={xAxisTicks}
                            tickMargin={12}
                        />
                        <YAxis
                            label={{ value: yAxisLabel, angle: -90, position: 'center', dx: -30 }}
                            domain={yAxisDomain}
                            ticks={yAxisTicks}
                            tickMargin={12}
                        />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#fff', border: '1px solid #ccc' }}
                            labelStyle={{ color: '#333' }}
                        />
                        <Area
                            type='linear'
                            dataKey={dataKey}
                            name={yAxisLabel}
                            stroke='#00C7FD'
                            strokeWidth={2}
                            fill='#00C7FD'
                            fillOpacity={0.3}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </View>
        </Flex>
    );
};
