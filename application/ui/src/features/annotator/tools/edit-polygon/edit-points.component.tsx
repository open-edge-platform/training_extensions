// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from 'react';

import { isEmpty } from 'lodash-es';

import { useEventListener } from '../../../../hooks/event-listener.hook';
import { ResizeAnchor } from '../../../../shared/annotator/resize-anchor.component';
import { isKeyboardDelete } from '../utils';
import { ResizeAnchorsGhostPoint } from './resize-anchors-ghost-point.component';
import { EditPointsProps, selectAnchorPointLabel } from './utils';

export const EditPoints = ({ zoom, shape, addPoint, onComplete, moveAnchorTo, removePoints }: EditPointsProps) => {
    const containerRef = useRef<SVGGElement | null>(null);
    const ref = useRef<SVGRectElement>(null);

    const [selectedAnchorIndexes, setSelectedAnchorIndexes] = useState<number[]>([]);

    useEventListener('keydown', (event: KeyboardEvent) => {
        if (isKeyboardDelete(event)) {
            event.preventDefault();

            handleRemovePoints();
        }
    });

    useEffect(() => {
        // Reset the selected anchors every time we add or remove a point
        setSelectedAnchorIndexes([]);
    }, [shape.points.length]);

    const handleRemovePoints = () => {
        removePoints(selectedAnchorIndexes);
        setSelectedAnchorIndexes([]);
    };

    const selectAnchorPoint = (idx: number, shiftKey: boolean, isContextMenu = false) => {
        setSelectedAnchorIndexes((indexes) => {
            if (isEmpty(indexes) || !shiftKey) {
                return [idx];
            }

            const isExistingIndex = indexes.includes(idx);

            // if shift key was pressed, toggle selection
            if (!isContextMenu && isExistingIndex) {
                return indexes.filter((otherIdx) => otherIdx !== idx);
            }

            if (isContextMenu && isExistingIndex) {
                return [...indexes];
            }

            return [...indexes, idx];
        });
    };

    return (
        <g style={{ pointerEvents: 'auto' }} ref={containerRef}>
            {/* Required to get correct relative mouse point */}
            <rect x={0} y={0} pointerEvents='none' fillOpacity={0} ref={ref} />

            <ResizeAnchorsGhostPoint
                svgRef={ref}
                moveAnchorTo={moveAnchorTo}
                shape={shape}
                addPoint={addPoint}
                onComplete={onComplete}
                zoom={zoom}
            />
            {shape.points.map((point, idx) => {
                const isSelected = selectedAnchorIndexes.includes(idx);
                const label = selectAnchorPointLabel(idx, isSelected, selectedAnchorIndexes);

                return (
                    <g
                        key={idx}
                        onPointerDown={(event) => {
                            event.stopPropagation();
                            selectAnchorPoint(idx, event.shiftKey);
                        }}
                        onClick={(event) => {
                            event.stopPropagation();
                        }}
                        aria-label={label}
                        aria-selected={isSelected}
                        onContextMenu={(event) => {
                            // we don't want event to be caught by annotation context menu
                            event.stopPropagation();
                            selectAnchorPoint(idx, event.shiftKey, true);
                        }}
                    >
                        <ResizeAnchor
                            {...point}
                            zoom={zoom}
                            onComplete={onComplete}
                            fill={isSelected ? 'var(--energy-blue)' : undefined}
                            label={`Resize polygon ${idx} anchor`}
                            moveAnchorTo={(x: number, y: number) => moveAnchorTo(idx, x, y)}
                        />
                    </g>
                );
            })}
        </g>
    );
};
