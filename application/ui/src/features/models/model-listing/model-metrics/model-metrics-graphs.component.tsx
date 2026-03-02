// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Text } from '@geti/ui';

import { LineMetric } from '../../../../constants/shared-types';
import { MetricGraph } from './metric-graph.component';

type ModelMetricsGraphsProps = {
    trainingMetrics: LineMetric[];
};

type GraphPoint = { x: number; y: number };

type GraphData = {
    key: string;
    title: string;
    xAxisLabel: string;
    yAxisLabel: string;
    data: GraphPoint[];
};

const MAX_X_AXIS_TICKS = 8;

const getSampledTicks = (points: { x: number; y: number }[]): number[] => {
    if (points.length <= MAX_X_AXIS_TICKS) {
        return points.map(({ x }) => x);
    }

    const stride = (points.length - 1) / (MAX_X_AXIS_TICKS - 1);

    return Array.from({ length: MAX_X_AXIS_TICKS }, (_, index) => {
        const pointIndex = Math.round(index * stride);

        return points[pointIndex].x;
    });
};

const normalizeGraphPoints = (points: GraphPoint[]): GraphPoint[] => {
    const pointsByX = new Map<number, number>();

    points.forEach(({ x, y }) => {
        if (!Number.isFinite(x) || !Number.isFinite(y)) {
            return;
        }

        pointsByX.set(x, y);
    });

    return Array.from(pointsByX.entries())
        .sort(([leftX], [rightX]) => leftX - rightX)
        .map(([x, y]) => ({ x, y }));
};

const formatMetricValue = (value: number): string => {
    if (value >= 0 && value <= 1) {
        return `${(value * 100).toFixed(2)}%`;
    }

    return value.toFixed(4);
};

const shouldRenderAsGraph = (data: GraphPoint[]): boolean => {
    if (data.length < 2) {
        return false;
    }

    const uniqueXCount = new Set(data.map(({ x }) => x)).size;

    return uniqueXCount >= 2;
};

export const ModelMetricsGraphs = ({ trainingMetrics }: ModelMetricsGraphsProps) => {
    const graphMap = trainingMetrics.reduce<Map<string, GraphData>>((accumulator, metric) => {
        const graphKey = metric.key;
        const existingGraph = accumulator.get(graphKey);
        const points = metric.value.line_data.flatMap((line) =>
            line.points.map((point) => ({ x: point.x, y: point.y }))
        );

        if (existingGraph === undefined) {
            accumulator.set(graphKey, {
                key: graphKey,
                title: metric.header,
                xAxisLabel: metric.value.x_axis_label,
                yAxisLabel: metric.value.y_axis_label,
                data: points,
            });

            return accumulator;
        }

        existingGraph.data.push(...points);

        return accumulator;
    }, new Map<string, GraphData>());

    const graphs = Array.from(graphMap.values()).map((graph) => {
        const data = normalizeGraphPoints(graph.data);

        return {
            ...graph,
            data,
            xAxisTicks: getSampledTicks(data),
        };
    });

    const valueMetrics = graphs
        .filter((metric) => !shouldRenderAsGraph(metric.data))
        .map((metric) => ({
            key: metric.key,
            title: metric.title,
            value: metric.data[metric.data.length - 1]?.y,
        }))
        .filter((metric): metric is { key: string; title: string; value: number } => metric.value !== undefined);

    const lineMetrics = graphs.filter((metric) => shouldRenderAsGraph(metric.data));

    return (
        <Flex width={'100%'} direction={'column'} gap={'size-300'}>
            {valueMetrics.length > 0 && (
                <Flex alignItems={'center'} gap={'size-300'} wrap>
                    {valueMetrics.map((metric) => (
                        <Text key={metric.key}>{`${metric.title}: ${formatMetricValue(metric.value)}`}</Text>
                    ))}
                </Flex>
            )}

            {lineMetrics.length > 0 && (
                <Flex width={'100%'} direction={'row'} gap={'size-300'} wrap>
                    {lineMetrics.map((graph) => (
                        <MetricGraph
                            key={graph.key}
                            title={graph.title}
                            data={graph.data}
                            dataKey='y'
                            xAxisKey='x'
                            xAxisLabel={graph.xAxisLabel}
                            yAxisLabel={graph.yAxisLabel}
                            xAxisTicks={graph.xAxisTicks}
                        />
                    ))}
                </Flex>
            )}
        </Flex>
    );
};
