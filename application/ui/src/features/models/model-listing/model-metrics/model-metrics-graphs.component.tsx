// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';

import { LineMetric } from '../../../../constants/shared-types';
import { MetricGraph } from './metric-graph.component';

type ModelMetricsGraphsProps = {
    trainingMetrics: LineMetric[];
};

export const ModelMetricsGraphs = ({ trainingMetrics }: ModelMetricsGraphsProps) => {
    const graphs = trainingMetrics.map((metric) => {
        const line = metric.value.line_data[0];
        const data = line?.points.map((point) => ({ x: point.x, y: point.y })) ?? [];
        const xAxisTicks = data.map(({ x }) => x);

        return {
            title: metric.header,
            xAxisLabel: metric.value.x_axis_label,
            yAxisLabel: metric.value.y_axis_label,
            data,
            xAxisTicks,
        };
    });

    return (
        <Flex width={'100%'} direction={'row'} gap={'size-300'} wrap>
            {graphs.map((graph) => (
                <MetricGraph
                    key={graph.title}
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
    );
};
