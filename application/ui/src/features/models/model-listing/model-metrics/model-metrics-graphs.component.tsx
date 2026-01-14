// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Flex, Heading, Item, Menu, MenuTrigger, View } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

type ModelMetricsGraphsProps = {
    // TODO: update types
    lossData?: { epoch: number; trainLoss: number }[];
    accuracyData?: { epoch: number; trainAccuracy: number }[];
};

export const ModelMetricsGraphs = ({ lossData, accuracyData }: ModelMetricsGraphsProps) => {
    return (
        <Flex width={'100%'} direction={'row'} gap={'size-300'}>
            <Flex flex={1} direction={'column'}>
                <Flex
                    justifyContent={'space-between'}
                    alignItems={'center'}
                    UNSAFE_style={{
                        backgroundColor: 'var(--spectrum-global-color-gray-200)',
                        padding: 'var(--spectrum-global-dimension-size-50) var(--spectrum-global-dimension-size-200)',
                    }}
                >
                    <Heading level={5}>ACCURACY</Heading>
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
                <View paddingY={'size-200'} paddingX={'size-550'} backgroundColor={'gray-50'}>
                    <ResponsiveContainer width='100%' height={268}>
                        <AreaChart data={accuracyData} margin={{ top: 35, bottom: 35, left: 20 }}>
                            <CartesianGrid />
                            <XAxis
                                dataKey='epoch'
                                label={{ value: 'epoch', position: 'bottom', fill: '#666', offset: 12 }}
                                ticks={[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 43]}
                                tickMargin={12}
                            />
                            <YAxis
                                label={{ value: 'train accuracy', angle: -90, position: 'center', dx: -30 }}
                                domain={[0, 1]}
                                ticks={[0, 0.2, 0.4, 0.6, 0.8, 1.0]}
                                tickMargin={12}
                            />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#fff', border: '1px solid #ccc' }}
                                labelStyle={{ color: '#333' }}
                            />
                            <Area
                                type='linear'
                                dataKey='trainAccuracy'
                                name='train accuracy'
                                stroke='#00C7FD'
                                strokeWidth={2}
                                fill='#00C7FD'
                                fillOpacity={0.3}
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                </View>
            </Flex>

            <Flex flex={1} direction={'column'}>
                <Flex
                    justifyContent={'space-between'}
                    alignItems={'center'}
                    UNSAFE_style={{
                        backgroundColor: 'var(--spectrum-global-color-gray-200)',
                        padding: 'var(--spectrum-global-dimension-size-50) var(--spectrum-global-dimension-size-200)',
                    }}
                >
                    <Heading level={5}>LOSS</Heading>
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
                <View paddingY={'size-200'} paddingX={'size-550'} backgroundColor={'gray-50'}>
                    <ResponsiveContainer width='100%' height={268}>
                        <AreaChart data={lossData} margin={{ top: 35, bottom: 35, left: 20 }}>
                            <CartesianGrid />
                            <XAxis
                                dataKey='epoch'
                                label={{ value: 'epoch', position: 'bottom', fill: '#666', offset: 12 }}
                                ticks={[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 43]}
                                tickMargin={12}
                            />
                            <YAxis
                                label={{ value: 'train loss', angle: -90, position: 'center', dx: -30 }}
                                tickMargin={12}
                            />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#fff', border: '1px solid #ccc' }}
                                labelStyle={{ color: '#333' }}
                            />
                            <Area
                                type='linear'
                                dataKey='trainLoss'
                                name='train loss'
                                stroke='#00C7FD'
                                strokeWidth={2}
                                fill='#00C7FD'
                                fillOpacity={0.3}
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                </View>
            </Flex>
        </Flex>
    );
};
