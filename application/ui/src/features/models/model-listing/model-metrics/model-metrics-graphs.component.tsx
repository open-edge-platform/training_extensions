// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Flex, Grid, Loading, minmax, repeat } from '@geti-ui/ui';
import { useIsVisible } from 'hooks/use-is-visible.hook';

import type { LineMetric } from '../../../../constants/shared-types';
import { Box } from '../components/box/box.component';
import { MetricGraph, type MetricGraphPoint } from './metric-graph.component';

import classes from './model-metrics.graphs.module.scss';

type ModelMetricsGraphsProps = {
    trainingMetrics: LineMetric[];
};

type GraphData = {
    key: string;
    title: string;
    xAxisLabel: string;
    yAxisLabel: string;
    data: MetricGraphPoint[];
};

const GraphPlaceholder = ({ title }: { title: string }) => (
    <Flex UNSAFE_className={classes.graphContainer}>
        <Box
            title={title}
            content={
                <Flex
                    alignItems={'center'}
                    justifyContent={'center'}
                    minHeight={'size-3000'}
                    UNSAFE_style={{ backgroundColor: 'var(--spectrum-gray-50)' }}
                >
                    <Loading mode={'inline'} size={'L'} />
                </Flex>
            }
        />
    </Flex>
);

const LazyMetricGraph = ({ graph }: { graph: GraphData }) => {
    const [container, setContainer] = useState<HTMLDivElement | null>(null);

    const isVisible = useIsVisible({ element: container });

    return (
        <div ref={setContainer} className={classes.graphContainer}>
            {isVisible ? (
                <MetricGraph
                    title={graph.title}
                    data={graph.data}
                    xAxisLabel={graph.xAxisLabel}
                    yAxisLabel={graph.yAxisLabel}
                />
            ) : (
                <GraphPlaceholder title={graph.title} />
            )}
        </div>
    );
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
        <Grid columns={repeat('auto-fit', minmax('size-6000', '1fr'))} gap={'size-300'}>
            {graphs.map((graph) => (
                <LazyMetricGraph key={graph.key} graph={graph} />
            ))}
        </Grid>
    );
};
