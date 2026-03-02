// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';

import type { LineMetric } from '../../../../constants/shared-types';
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

export const ModelMetricsGraphs = ({ trainingMetrics }: ModelMetricsGraphsProps) => {
    const graphs: GraphData[] = trainingMetrics.map((metric) => ({
        key: metric.key,
        title: metric.header,
        xAxisLabel: metric.value.x_axis_label,
        yAxisLabel: metric.value.y_axis_label,
        data: metric.value.line_data.flatMap((line) => line.points.map((point) => ({ x: point.x, y: point.y }))),
    }));

    return (
        <Flex width={'100%'} direction={'row'} gap={'size-300'} wrap>
            {graphs.map((graph) => (
                <MetricGraph
                    key={graph.key}
                    title={graph.title}
                    data={graph.data}
                    dataKey='y'
                    xAxisKey='x'
                    xAxisLabel={graph.xAxisLabel}
                    yAxisLabel={graph.yAxisLabel}
                />
            ))}
        </Flex>
    );
};
