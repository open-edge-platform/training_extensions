// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useRef, useState } from 'react';

import { Flex } from '@geti/ui';

import type { LineMetric } from '../../../../constants/shared-types';
import { MetricGraph, type MetricGraphPoint } from './metric-graph.component';

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

const useProgressiveList = <T,>(items: T[]): T[] => {
    const [visible, setVisible] = useState<T[]>([]);
    const frameRef = useRef<number | null>(null);

    useEffect(() => {
        setVisible([]);
        if (items.length === 0) {
            return;
        }

        let index = 1;

        const renderNext = () => {
            setVisible(items.slice(0, index));

            if (index >= items.length) {
                frameRef.current = null;

                return;
            }

            index += 1;
            frameRef.current = requestAnimationFrame(renderNext);
        };

        frameRef.current = requestAnimationFrame(renderNext);

        return () => {
            if (frameRef.current !== null) {
                cancelAnimationFrame(frameRef.current);
                frameRef.current = null;
            }
        };
    }, [items]);

    return visible;
};

export const ModelMetricsGraphs = ({ trainingMetrics }: ModelMetricsGraphsProps) => {
    const graphs = useMemo<GraphData[]>(
        () =>
            trainingMetrics.map((metric) => ({
                key: metric.key,
                title: metric.header,
                xAxisLabel: metric.value.x_axis_label,
                yAxisLabel: metric.value.y_axis_label,
                data: metric.value.line_data.flatMap((line) =>
                    line.points.map((point) => ({ x: point.x, y: point.y }))
                ),
            })),
        [trainingMetrics]
    );

    const visibleGraphs = useProgressiveList(graphs);

    return (
        <Flex width={'100%'} direction={'row'} gap={'size-300'} wrap>
            {visibleGraphs.map((graph) => (
                <MetricGraph
                    key={graph.key}
                    title={graph.title}
                    data={graph.data}
                    xAxisLabel={graph.xAxisLabel}
                    yAxisLabel={graph.yAxisLabel}
                />
            ))}
        </Flex>
    );
};
