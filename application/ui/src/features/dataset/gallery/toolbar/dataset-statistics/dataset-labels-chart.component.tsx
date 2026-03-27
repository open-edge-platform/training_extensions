// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue } from '@geti/ui';
import { useProject } from 'hooks/api/project.hook';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, XAxis, YAxis } from 'recharts';

type DatasetLabelsChartProps = {
    totalItems: number;
    instancesPerLabel: {
        label_id: string;
        instances: number;
    }[];
};

const getAxisTicks = (total: number, step: number): number[] => {
    const ticks = Array.from({ length: Math.floor(total / step) + 1 }, (_, i) => i * step);

    return total % step !== 0 ? [...ticks, total] : ticks;
};

export const DatasetLabelsChart = ({ totalItems, instancesPerLabel }: DatasetLabelsChartProps) => {
    const { data: selectedProject } = useProject();
    const projectLabels = selectedProject?.task?.labels ?? [];

    const chartData = instancesPerLabel.map((item) => ({
        label: projectLabels.find((label) => label.id === item.label_id)?.name ?? item.label_id,
        score: item.instances,
    }));

    return (
        <ResponsiveContainer width='100%' height={'100%'} minHeight={200}>
            <BarChart
                data={chartData}
                layout='vertical'
                margin={{ top: 10, right: 30, left: 10, bottom: 10 }}
                barCategoryGap={20}
            >
                <CartesianGrid stroke='var(--spectrum-global-color-gray-600)' strokeOpacity={0.4} horizontal={false} />

                <XAxis
                    type='number'
                    domain={[0, totalItems]}
                    ticks={getAxisTicks(totalItems, 20)}
                    tick={{ fill: 'var(--spectrum-global-color-gray-800)', fontSize: dimensionValue('size-200') }}
                    axisLine={{ stroke: 'var(--spectrum-global-color-gray-600)', strokeWidth: 1 }}
                    tickLine={false}
                />

                <YAxis
                    type='category'
                    dataKey='label'
                    width={140}
                    tick={{ fill: 'var(--spectrum-global-color-gray-800)', fontSize: dimensionValue('size-200') }}
                    axisLine={{ stroke: 'var(--spectrum-global-color-gray-600)', strokeWidth: 1 }}
                    tickLine={false}
                />

                <Bar dataKey='score' fill='var(--moss)' radius={[4, 4, 4, 4]} barSize={36} />
            </BarChart>
        </ResponsiveContainer>
    );
};
