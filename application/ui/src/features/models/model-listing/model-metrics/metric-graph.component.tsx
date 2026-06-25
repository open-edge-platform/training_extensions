// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useRef } from 'react';

import { ActionButton, Flex, View } from '@geti/ui';
import { DownloadIcon } from '@geti/ui/icons';
import { CartesianGrid, Line, LineChart, Tooltip, XAxis, YAxis } from 'recharts';

import { Box } from '../components/box/box.component';

export type MetricGraphPoint = {
    x: number;
    y: number;
};

type MetricGraphProps = {
    title: string;
    data?: MetricGraphPoint[];
    xAxisLabel?: string;
    yAxisLabel: string;
};

const X_AXIS_TICK_COUNT = 8;
const Y_AXIS_TICK_COUNT = 4;

export const MetricGraph = ({ title, data, xAxisLabel, yAxisLabel }: MetricGraphProps) => {
    const graphRef = useRef<HTMLDivElement>(null);

    const handleDownload = useCallback(() => {
        if (!graphRef.current) return;

        const svgElement = graphRef.current.querySelector('svg');
        if (!svgElement) return;

        // Clone the SVG so we don't modify the original
        const clonedSvg = svgElement.cloneNode(true) as SVGSVGElement;

        // Ensure inline styles are applied correctly by setting explicit width/height
        const { width, height } = svgElement.getBoundingClientRect();
        clonedSvg.setAttribute('width', `${width}`);
        clonedSvg.setAttribute('height', `${height}`);

        const svgString = new XMLSerializer().serializeToString(clonedSvg);
        const blob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
        const url = URL.createObjectURL(blob);

        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;

            const ctx = canvas.getContext('2d');
            if (ctx) {
                // White background
                ctx.fillStyle = '#ffffff';
                ctx.fillRect(0, 0, width, height);
                ctx.drawImage(img, 0, 0);

                const dataUrl = canvas.toDataURL('image/png');
                const link = document.createElement('a');
                link.href = dataUrl;
                link.download = `${title.replace(/\s+/g, '_').toLowerCase()}_metrics.png`;
                link.click();
            }

            URL.revokeObjectURL(url);
        };
        img.src = url;
    }, [title]);

    return (
        <Flex flex={1} direction={'column'} minWidth={'size-5000'}>
            <Box
                title={title}
                actions={
                    <ActionButton isQuiet onPress={handleDownload} aria-label={`Download ${title} graph`}>
                        <DownloadIcon />
                    </ActionButton>
                }
                content={
                    <View ref={graphRef} backgroundColor={'gray-50'} minHeight={'size-3800'}>
                        <LineChart
                            responsive
                            width={'100%'}
                            style={{ aspectRatio: 1.6 }}
                            data={data}
                            margin={{ top: 35, bottom: 35, left: 35 }}
                        >
                            <CartesianGrid />
                            <XAxis
                                dataKey='x'
                                type='number'
                                domain={[0, 'auto']}
                                label={{ value: xAxisLabel ?? 'x', position: 'bottom', fill: '#666', offset: 12 }}
                                tickCount={X_AXIS_TICK_COUNT}
                                tickMargin={12}
                            />
                            <YAxis
                                label={{ value: yAxisLabel, angle: -90, position: 'center', dx: -38, fill: '#666' }}
                                tickCount={Y_AXIS_TICK_COUNT}
                                tickMargin={12}
                                tickFormatter={(value) => Number(value).toFixed(4)}
                            />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#fff', border: '1px solid #ccc' }}
                                labelStyle={{ color: '#333' }}
                            />
                            <Line
                                type='linear'
                                dataKey='y'
                                name={yAxisLabel}
                                stroke='var(--energy-blue)'
                                strokeWidth={2}
                                dot={false}
                            />
                        </LineChart>
                    </View>
                }
            />
        </Flex>
    );
};
