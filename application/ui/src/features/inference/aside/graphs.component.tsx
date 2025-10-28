// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from 'react';

import { Content, Flex, Grid, Heading, IllustratedMessage, View } from '@geti/ui';
import { usePipelineMetrics } from 'hooks/api/pipeline.hook';
import { CartesianGrid, Label, Line, LineChart, ReferenceLine, XAxis, YAxis } from 'recharts';

interface DataPoint {
    name: string;
    value: number;
}

const MAX_DATA_POINTS = 60; // Keep last 60 data points

const useMetricsData = () => {
    const [latencyData, setLatencyData] = useState<DataPoint[]>([]);
    const [throughputData, setThroughputData] = useState<DataPoint[]>([]);
    const counterRef = useRef(0);

    const { data: metrics } = usePipelineMetrics();

    useEffect(() => {
        if (!metrics) return;

        const dataPointName = `${counterRef.current++}`;

        setLatencyData((prev) => {
            const newData = [
                ...prev,
                {
                    name: dataPointName,
                    value: metrics.inference.latency.avg_ms ?? 0,
                },
            ];

            // Keep only last MAX_DATA_POINTS
            return newData.slice(-MAX_DATA_POINTS);
        });

        setThroughputData((prev) => {
            const newData = [
                ...prev,
                {
                    name: dataPointName,
                    value: metrics.inference.throughput.avg_requests_per_second ?? 0,
                },
            ];
            return newData.slice(-MAX_DATA_POINTS);
        });
    }, [metrics]);

    return { latencyData, throughputData, metrics };
};

const Graph = ({ label, data }: { label: string; data: DataPoint[] }) => {
    return (
        <View padding={{ top: 'size-250', left: 'size-200', right: 'size-200', bottom: 'size-125' }}>
            <LineChart width={500} height={200} data={data}>
                <XAxis
                    minTickGap={32}
                    stroke='var(--spectrum-global-color-gray-800)'
                    dataKey='name'
                    tickLine={false}
                    tickMargin={8}
                />
                <YAxis
                    tickLine={false}
                    stroke='var(--spectrum-global-color-gray-900)'
                    dataKey='value'
                    tickFormatter={(value: number) => {
                        return value > 10 ? value.toFixed(0) : value.toFixed(2);
                    }}
                >
                    <Label
                        angle={-90}
                        value={label}
                        position='insideLeft'
                        offset={10}
                        style={{
                            textAnchor: 'middle',
                            fill: 'var(--spectrum-global-color-gray-900)',
                            fontSize: '10px',
                        }}
                    />
                </YAxis>
                <CartesianGrid stroke='var(--spectrum-global-color-gray-400)' />
                {data.length > 0 && (
                    <ReferenceLine x={data[0].name} stroke='var(--spectrum-global-color-gray-600)' strokeWidth={2} />
                )}
                <Line
                    type='linear'
                    dataKey='value'
                    dot={false}
                    stroke='var(--energy-blue)'
                    isAnimationActive={false}
                    strokeWidth='3'
                />
            </LineChart>
        </View>
    );
};

export const Graphs = () => {
    const { latencyData, throughputData, metrics } = useMetricsData();

    const hasData = latencyData.length > 0 || throughputData.length > 0;

    return (
        <Grid
            gridArea={'aside'}
            height={'90vh'}
            areas={['header', 'graphs']}
            rows={['min-content', 'minmax(0, 1fr)']}
            UNSAFE_style={{
                padding: 'var(--spectrum-global-dimension-size-200)',
            }}
        >
            <Flex gridArea={'header'} alignItems='center' gap={'size-100'} marginBottom={'size-300'}>
                <Heading level={4}>Model statistics</Heading>
            </Flex>
            <View gridArea={'graphs'} UNSAFE_style={{ overflow: 'hidden auto' }}>
                {!hasData && !metrics ? (
                    <IllustratedMessage>
                        <Heading>No statistics available</Heading>
                        <Content>
                            Model statistics will appear here once the pipeline starts running and starts processing
                            data.
                        </Content>
                    </IllustratedMessage>
                ) : (
                    <>
                        <View paddingY='size-200'>
                            <Heading level={4} marginBottom={'size-300'}>
                                Throughput
                            </Heading>
                            <Graph label='requests/sec' data={throughputData} />
                        </View>
                        <View paddingY='size-200'>
                            <Heading level={4} marginBottom={'size-300'}>
                                Latency
                            </Heading>
                            <Graph label='ms' data={latencyData} />
                        </View>
                    </>
                )}
            </View>
        </Grid>
    );
};
