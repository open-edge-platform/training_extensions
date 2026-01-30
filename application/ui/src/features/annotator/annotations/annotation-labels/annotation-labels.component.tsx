// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CSSProperties, PointerEvent, useCallback } from 'react';

import { isEmpty } from 'lodash-es';
import { v4 as uuid } from 'uuid';

import { useZoom } from '../../../../components/zoom/zoom.provider';
import { AnnotationLabel } from '../../../../shared/types';
import { isPrediction } from '../utils';

import classes from '../annotation-labels.module.scss';

const labelStyles = (scale: number) => {
    // We need the actual values for calculations:
    // spectrum-global-dimension-size-50 = 4px
    // spectrum-global-dimension-size-100 = 8px
    // spectrum-global-dimension-size-175 = 14px
    // spectrum-global-dimension-size-250 = 20px
    // spectrum-global-dimension-size-300 = 24px

    const height = 24 / scale;
    const padding = 8 / scale;
    const closeButtonWidth = 20 / scale;
    const fontSize = 14 / scale;
    // Stroke width is 3px/scale, half extends outside the shape boundary (1.5px/scale)
    const strokeOffset = 1.5 / scale;
    // To prevent gaps between labels and box stroke due to anti-aliasing, we overlap by 0.2px/scale
    const yOffset = -height - 1.3 / scale;
    const charWidth = fontSize * 0.45;
    const borderRadius = 4 / scale;
    const textYOffset = 16 / scale; // Vertical centering offset for text
    const closeButtonXOffset = 15 / scale; // X offset for close button text

    return {
        height,
        padding,
        closeButtonWidth,
        fontSize,
        yOffset,
        charWidth,
        borderRadius,
        textYOffset,
        closeButtonXOffset,
        strokeOffset,
    };
};

const placeholderLabel = { id: uuid(), name: 'No label', color: 'var(--annotation-fill)', isPrediction: false };

interface AnnotationLabelsProps {
    labels: AnnotationLabel[];
    onRemove: (labelId: string) => void;
    useBottomCorners?: boolean;
}

const formatPredictionScore = (score: number) => {
    return new Intl.NumberFormat('en-US', { style: 'percent' }).format(score);
};

export const AnnotationLabels = ({ labels, onRemove, useBottomCorners = false }: AnnotationLabelsProps) => {
    const { scale } = useZoom();

    const styles = labelStyles(scale);
    const { height, padding, closeButtonWidth, fontSize, yOffset, charWidth } = styles;

    const getLabelText = (label: AnnotationLabel) => {
        return `${label.name} ${isPrediction(label) ? formatPredictionScore(label.probability) : ''}`.trim();
    };

    const calculateLabelWidth = (text: string) => {
        const textWidth = text.length * charWidth;
        return textWidth + padding * 2;
    };

    // Creates a path for a rectangle with optional rounded corners
    const createLabelPath = (
        x: number,
        y: number,
        width: number,
        h: number,
        topLeftRadius: number,
        topRightRadius: number,
        bottomRightRadius: number,
        bottomLeftRadius: number
    ) => {
        const r1 = topLeftRadius;
        const r2 = topRightRadius;
        const r3 = bottomRightRadius;
        const r4 = bottomLeftRadius;

        // Start at bottom-left (after curve), go clockwise
        return [
            `M ${x} ${y + h - r4}`, // Start on left edge above bottom-left curve
            `L ${x} ${y + r1}`, // Left edge up to top-left curve
            r1 > 0 ? `Q ${x} ${y} ${x + r1} ${y}` : `L ${x} ${y}`, // Top-left corner
            `L ${x + width - r2} ${y}`, // Top edge
            r2 > 0 ? `Q ${x + width} ${y} ${x + width} ${y + r2}` : `L ${x + width} ${y}`, // Top-right corner
            `L ${x + width} ${y + h - r3}`, // Right edge
            // Bottom-right corner
            r3 > 0 ? `Q ${x + width} ${y + h} ${x + width - r3} ${y + h}` : `L ${x + width} ${y + h}`,
            `L ${x + r4} ${y + h}`, // Bottom edge
            r4 > 0 ? `Q ${x} ${y + h} ${x} ${y + h - r4}` : `L ${x} ${y + h}`, // Bottom-left corner
            'Z', // Close path
        ].join(' ');
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

    let fullLengthOfAllLabels = -styles.strokeOffset;

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

    return labels.map((label, index) => {
        const labelText = getLabelText(label);
        const labelWidth = !isEmpty(label.name) ? calculateLabelWidth(labelText) + closeButtonWidth : 0;
        const xOffset = fullLengthOfAllLabels;

        const isFirst = index === 0;
        const isLast = index === labels.length - 1;

        // Build border radius based on position: top-left for first, top-right for last
        // When useBottomCorners is true, also round bottom-left for first and bottom-right for last
        const { borderRadius } = styles;
        const topLeftRadius = isFirst ? borderRadius : 0;
        const topRightRadius = isLast ? borderRadius : 0;
        const bottomRightRadius = useBottomCorners && isLast ? borderRadius : 0;
        const bottomLeftRadius = useBottomCorners && isFirst ? borderRadius : 0;

        fullLengthOfAllLabels += labelWidth;

        return (
            <g
                key={label.id}
                fill='none'
                stroke='none'
                fillOpacity={1}
                style={{ '--labelColor': label.color } as CSSProperties}
                className={classes.labelContainer}
            >
                {/* Label name */}
                <path
                    d={createLabelPath(
                        xOffset,
                        yOffset,
                        labelWidth,
                        height,
                        topLeftRadius,
                        topRightRadius,
                        bottomRightRadius,
                        bottomLeftRadius
                    )}
                    fill={label.color}
                    aria-label={`label ${label.name} background`}
                />
                <text
                    x={xOffset + padding}
                    y={yOffset + styles.textYOffset}
                    fontSize={fontSize}
                    aria-label={`label ${label.name}`}
                >
                    {labelText}
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
                        aria-label={`Remove ${label.name}`}
                    >
                        ×
                    </text>
                </g>
            </g>
        );
    });
};
