// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useZoom } from 'src/components/zoom/zoom.provider';

import { Label } from '../types';

interface AnnotationLabelsProps {
    labels: Label[];
    onRemove: (labelId: string) => void;
}

export const AnnotationLabels = ({ labels, onRemove }: AnnotationLabelsProps) => {
    const height = 24; // spectrum-global-dimension-size-300
    const { scale } = useZoom();

    const scaledHeight = height / scale;
    const scaledWidth = 100 / scale;
    const yOffset = -scaledHeight;
    const fontSize = 14 / scale;
    const gap = 4 / scale; // Gap between labels

    const onDeleteLabel = (labelId: string) => (event: React.PointerEvent) => {
        event.preventDefault();
        event.stopPropagation();

        onRemove(labelId);
    };

    return labels.map((label, index) => {
        const xOffset = index * (scaledWidth + gap);

        return (
            <g key={label.id} fill='none' stroke='none' fillOpacity={1}>
                <rect
                    x={xOffset}
                    y={yOffset}
                    width={scaledWidth}
                    height={scaledHeight}
                    fill={label.color}
                    stroke='none'
                    rx={4 / scale}
                />
                <text x={xOffset + 8 / scale} y={yOffset + 16 / scale} fontSize={fontSize} fill='#fff'>
                    {label.name}
                </text>

                <g style={{ cursor: 'pointer', pointerEvents: 'auto' }} onPointerDown={onDeleteLabel(label.id)}>
                    <rect
                        x={xOffset + scaledWidth - 20 / scale}
                        y={yOffset}
                        width={20 / scale}
                        height={scaledHeight}
                        fill='transparent'
                    />
                    <text
                        x={xOffset + scaledWidth - 12 / scale}
                        y={yOffset + 16 / scale}
                        fontSize={fontSize}
                        fill='#fff'
                    >
                        x
                    </text>
                </g>
            </g>
        );
    });
};
