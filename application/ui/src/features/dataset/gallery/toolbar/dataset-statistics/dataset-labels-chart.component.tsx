// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue } from '@geti-ui/ui';
import {
    Bar,
    BarChart,
    CartesianGrid,
    Label,
    LabelList,
    LabelProps,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts';

import { isEmptyLabel, useProjectLabelsWithEmptyLabel } from '../../../../../shared/annotator/labels';

type DatasetLabelsChartProps = {
    totalItems: number;
    instancesPerLabel: {
        label_id: string | null;
        instances: number;
    }[];
};

const BAR_SIZE = 36;
const MIN_CHART_HEIGHT = 192;

const getAxisTicks = (total: number): number[] => {
    const TICK_SPACING = 20;

    const ticks = Array.from({ length: Math.floor(total / TICK_SPACING) + 1 }, (_, i) => i * TICK_SPACING);

    return total % TICK_SPACING !== 0 ? [...ticks, total] : ticks;
};

const ItemLabel = (props: LabelProps & { labelColor?: string }) => {
    const { labelColor, value, ...rest } = props;
    return value === 0 ? null : (
        <Label
            {...rest}
            value={value}
            style={{
                fill: labelColor ? `lch(from ${labelColor} calc((50 - l) * infinity) 0 0)` : 'white',
            }}
        />
    );
};

export const DatasetLabelsChart = ({ totalItems, instancesPerLabel }: DatasetLabelsChartProps) => {
    const projectLabels = useProjectLabelsWithEmptyLabel();

    const emptyLabelInstance = instancesPerLabel.find(({ label_id }) => label_id === null);

    const chartData = projectLabels.map((projectLabel) => {
        const matchingInstances = isEmptyLabel(projectLabel)
            ? emptyLabelInstance
            : instancesPerLabel.find(({ label_id }) => label_id === projectLabel.id);

        return {
            label: projectLabel.name,
            color: projectLabel.color,
            score: matchingInstances?.instances ?? 0,
        };
    });

    return (
        <ResponsiveContainer
            width='100%'
            height='100%'
            minHeight={Math.max(projectLabels.length * BAR_SIZE, MIN_CHART_HEIGHT)}
        >
            <BarChart data={chartData} layout='vertical' margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
                <CartesianGrid stroke='var(--spectrum-global-color-gray-600)' strokeOpacity={0.4} horizontal={false} />

                <XAxis
                    type='number'
                    tickLine={false}
                    domain={[0, totalItems]}
                    ticks={getAxisTicks(totalItems)}
                    axisLine={{ stroke: 'var(--spectrum-global-color-gray-600)', strokeWidth: 1 }}
                    tick={{ fill: 'var(--spectrum-global-color-gray-800)', fontSize: dimensionValue('size-200') }}
                />

                <YAxis
                    type='category'
                    dataKey='label'
                    width={140}
                    interval={0}
                    tick={{ fill: 'var(--spectrum-global-color-gray-800)', fontSize: dimensionValue('size-200') }}
                    axisLine={{ stroke: 'var(--spectrum-global-color-gray-600)', strokeWidth: 1 }}
                    tickLine={false}
                />

                <Bar dataKey='score' radius={[4, 4, 4, 4]} fill={'color'} barSize={BAR_SIZE}>
                    <LabelList
                        dataKey='score'
                        position='insideEnd'
                        content={(props) => (
                            <ItemLabel {...props} labelColor={chartData?.at(props.index ?? 0)?.color} />
                        )}
                    />
                </Bar>

                <Tooltip
                    shared={false}
                    formatter={(value) => [value, 'Annotations']}
                    itemStyle={{ color: 'var(--spectrum-global-color-gray-800)' }}
                    contentStyle={{ background: 'var(--spectrum-global-color-gray-50)' }}
                />
            </BarChart>
        </ResponsiveContainer>
    );
};
