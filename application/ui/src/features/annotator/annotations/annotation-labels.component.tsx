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
    const yOffset = -scaledHeight;
    const fontSize = 14 / scale;
    const gap = 4 / scale; // Gap between labels
    const padding = 8 / scale; // Horizontal padding inside label
    const closeButtonWidth = 20 / scale; // Width of close button area
    const charWidth = fontSize * 0.6; // ~60% of font size for typical monospace-ish rendering

    const calculateLabelWidth = (text: string) => {
        const textWidth = text.length * charWidth;
        return textWidth + padding * 2 + closeButtonWidth;
    };

    const onDeleteLabel = (labelId: string) => (event: React.PointerEvent) => {
        event.preventDefault();
        event.stopPropagation();

        onRemove(labelId);
    };

    let fullLengthOfAllLabels = 0;

    return labels.map((label) => {
        const labelWidth = calculateLabelWidth(label.name);
        const xOffset = fullLengthOfAllLabels;

        fullLengthOfAllLabels += labelWidth + gap;

        return (
            <g key={label.id} fill='none' stroke='none' fillOpacity={1}>
                <rect
                    x={xOffset}
                    y={yOffset}
                    width={labelWidth}
                    height={scaledHeight}
                    fill={label.color}
                    stroke='none'
                    rx={4 / scale}
                />
                <text x={xOffset + padding} y={yOffset + 16 / scale} fontSize={fontSize} fill='#fff'>
                    {label.name}
                </text>

                <g style={{ cursor: 'pointer', pointerEvents: 'auto' }} onPointerDown={onDeleteLabel(label.id)}>
                    <rect
                        x={xOffset + labelWidth - closeButtonWidth}
                        y={yOffset}
                        width={closeButtonWidth}
                        height={scaledHeight}
                        fill='transparent'
                    />
                    <text
                        x={xOffset + labelWidth - 12 / scale}
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
