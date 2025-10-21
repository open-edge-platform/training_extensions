// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, useCallback } from 'react';

import { isEmpty } from 'lodash-es';
import { useZoom } from 'src/components/zoom/zoom.provider';
import { Label } from 'src/constants/shared-types';
import { v4 as uuid } from 'uuid';

const labelStyles = (scale: number) => {
    // We need the actual values for calculations:
    // spectrum-global-dimension-size-50 = 4px
    // spectrum-global-dimension-size-100 = 8px
    // spectrum-global-dimension-size-175 = 14px
    // spectrum-global-dimension-size-250 = 20px
    // spectrum-global-dimension-size-300 = 24px

    const height = 24 / scale;
    const padding = 8 / scale;
    const gap = 2 / scale; // Gap between labels
    const closeButtonWidth = 20 / scale;
    const fontSize = 14 / scale;
    const yOffset = -height;
    const charWidth = fontSize * 0.6;
    const borderRadius = 4 / scale;
    const textYOffset = 16 / scale; // Vertical centering offset for text
    const closeButtonXOffset = 12 / scale; // X offset for close button text

    return {
        height,
        padding,
        gap,
        closeButtonWidth,
        fontSize,
        yOffset,
        charWidth,
        borderRadius,
        textYOffset,
        closeButtonXOffset,
    };
};

const placeholderLabel = { id: uuid(), name: 'No label', color: 'var(--annotation-fill)', isPrediction: false };

interface AnnotationLabelsProps {
    labels: Label[];
    onRemove: (labelId: string) => void;
}
export const AnnotationLabels = ({ labels, onRemove }: AnnotationLabelsProps) => {
    const { scale } = useZoom();

    const styles = labelStyles(scale);
    const { height, padding, gap, closeButtonWidth, fontSize, yOffset, charWidth } = styles;

    const calculateLabelWidth = (text: string) => {
        const textWidth = text.length * charWidth;
        return textWidth + padding * 2;
    };

    const onDeleteLabel = useCallback(
        (labelId: string) => (event: PointerEvent) => {
            // To avoid triggering onPointerDown of the parent svg
            event.preventDefault();
            event.stopPropagation();

            onRemove(labelId);
        },
        [onRemove]
    );

    let fullLengthOfAllLabels = 0;

    if (!labels.length) {
        const placeholderLabelWidth = calculateLabelWidth(placeholderLabel.name);

        return (
            <g key={placeholderLabel.id} fill='none' stroke='none' fillOpacity={1}>
                {/* Label name */}
                <rect
                    x={0}
                    y={yOffset}
                    width={placeholderLabelWidth}
                    height={height}
                    fill={placeholderLabel.color}
                    stroke='none'
                    rx={styles.borderRadius}
                    aria-label={`label ${placeholderLabel.name} background`}
                />
                <text
                    x={padding}
                    y={yOffset + styles.textYOffset}
                    fontSize={fontSize}
                    fill='#fff'
                    aria-label={`label ${placeholderLabel.name}`}
                >
                    {placeholderLabel.name}
                </text>
            </g>
        );
    }

    return labels.map((label) => {
        const labelWidth = !isEmpty(label.name) ? calculateLabelWidth(label.name) + closeButtonWidth : 0;
        const xOffset = fullLengthOfAllLabels;

        fullLengthOfAllLabels += labelWidth + gap;

        return (
            <g key={label.id} fill='none' stroke='none' fillOpacity={1}>
                {/* Label name */}
                <rect
                    x={xOffset}
                    y={yOffset}
                    width={labelWidth}
                    height={height}
                    fill={label.color}
                    stroke='none'
                    rx={styles.borderRadius}
                    aria-label={`label ${label.name} background`}
                />
                <text
                    x={xOffset + padding}
                    y={yOffset + styles.textYOffset}
                    fontSize={fontSize}
                    fill='#fff'
                    aria-label={`label ${label.name}`}
                >
                    {label.name}
                </text>

                {/* Remove button */}
                <g style={{ cursor: 'pointer', pointerEvents: 'auto' }} onPointerDown={onDeleteLabel(label.id)}>
                    <rect
                        x={xOffset + labelWidth - closeButtonWidth}
                        y={yOffset}
                        width={closeButtonWidth}
                        height={height}
                        fill='transparent'
                    />
                    <text
                        x={xOffset + labelWidth - styles.closeButtonXOffset}
                        y={yOffset + styles.textYOffset}
                        fontSize={fontSize}
                        fill='#fff'
                        aria-label={`Remove ${label.name}`}
                    >
                        x
                    </text>
                </g>
            </g>
        );
    });
};
