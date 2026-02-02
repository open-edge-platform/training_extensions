// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, useCallback } from 'react';

import { v4 as uuid } from 'uuid';

import { useZoom } from '../../../../components/zoom/zoom.provider';
import { AnnotationLabel } from '../../../../shared/types';
import { isPrediction } from '../utils';

import classes from './annotation-labels.module.scss';

const placeholderLabel = { id: uuid(), name: 'No label', color: 'var(--annotation-fill)', isPrediction: false };

// Screen-space dimensions for the foreignObject hit area
const LABEL_HEIGHT_PX = 24;
const LABEL_MAX_WIDTH_PX = 1000;

interface AnnotationLabelsProps {
    labels: AnnotationLabel[];
    onRemove: (labelId: string) => void;
    useBottomCorners?: boolean;
}

const formatPredictionScore = (score: number) => {
    return new Intl.NumberFormat('en-US', { style: 'percent' }).format(score);
};

const getLabelText = (label: AnnotationLabel) => {
    return `${label.name} ${isPrediction(label) ? formatPredictionScore(label.probability) : ''}`.trim();
};

export const AnnotationLabels = ({ labels, onRemove, useBottomCorners = false }: AnnotationLabelsProps) => {
    const { scale } = useZoom();

    const onDeleteLabel = useCallback(
        (labelId: string) => (event: PointerEvent) => {
            event.preventDefault();
            event.stopPropagation();
            onRemove(labelId);
        },
        [onRemove]
    );

    const displayLabels = labels.length ? labels : [placeholderLabel];

    // Need to round up to preveent sub-pixel render issues when zoomed in
    const foreignObjectHeight = Math.ceil(LABEL_HEIGHT_PX / scale) + 1;
    const foreignObjectWidth = Math.ceil(LABEL_MAX_WIDTH_PX / scale) + 1;

    return (
        <foreignObject
            x={0}
            y={useBottomCorners ? 0 : -foreignObjectHeight}
            width={foreignObjectWidth}
            height={foreignObjectHeight}
            overflow='visible'
        >
            <div
                className={useBottomCorners ? classes.labelsContainerPolygon : classes.labelsContainerRect}
                style={{ height: '100%', alignItems: 'flex-end' }}
            >
                {displayLabels.map((label, index) => {
                    const isFirst = index === 0;
                    const isLast = index === displayLabels.length - 1;
                    const isPlaceholder = !labels.length;

                    return (
                        <div
                            key={label.id}
                            className={classes.label}
                            style={{
                                '--label-color': label.color,
                                '--border-top-left': isFirst ? '4px' : '0',
                                '--border-top-right': isLast ? '4px' : '0',
                                '--border-bottom-left': useBottomCorners && isFirst ? '4px' : '0',
                                '--border-bottom-right': useBottomCorners && isLast ? '4px' : '0',
                            }}
                            aria-label={`label ${label.name} background`}
                        >
                            <span aria-label={`label ${label.name}`}>{getLabelText(label)}</span>
                            {!isPlaceholder && (
                                <button
                                    className={classes.removeButton}
                                    onPointerDown={onDeleteLabel(label.id)}
                                    aria-label={`Remove ${label.name}`}
                                >
                                    ×
                                </button>
                            )}
                        </div>
                    );
                })}
            </div>
        </foreignObject>
    );
};
