// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, RefObject, useRef, useState } from 'react';

import { ANCHOR_SIZE } from '@geti/smart-tools';

import { ResizeAnchor } from '../../../../shared/annotator/resize-anchor.component';
import { isLeftButton } from '../../../../shared/buttons-utils';
import { Point } from '../../../../shared/types';
import { getRelativePoint, projectPointOnLine } from '../utils';
import { EditPointsProps } from './utils';

interface GhostPoint {
    idx: number;
    point: Point;
}

interface ResizeAnchorsProps
    extends Pick<EditPointsProps, 'shape' | 'moveAnchorTo' | 'addPoint' | 'onComplete' | 'zoom'> {
    svgRef: RefObject<SVGRectElement | null>;
}

export const ResizeAnchorsGhostPoint = ({
    shape,
    addPoint,
    moveAnchorTo,
    zoom,
    onComplete,
    svgRef,
}: ResizeAnchorsProps) => {
    const [ghostPoint, setGhostPoint] = useState<GhostPoint | undefined>(undefined);
    const ghostPointRef = useRef<GhostPoint | undefined>(undefined);

    const updateGhostPoint = (newGhost: GhostPoint | undefined) => {
        ghostPointRef.current = newGhost;
        setGhostPoint(newGhost);
    };

    const removeGhostPoint = () => {
        updateGhostPoint(undefined);
    };

    return (
        <>
            {shape.points.map((point, idx) => {
                const nextPoint = idx + 1 >= shape.points.length ? shape.points[0] : shape.points[idx + 1];

                const onPointerMove = (event: PointerEvent) => {
                    if (svgRef.current === null) {
                        return;
                    }

                    const mouse = getRelativePoint(svgRef.current, { x: event.clientX, y: event.clientY }, zoom);

                    const pointOnLine = projectPointOnLine([point, nextPoint], mouse);

                    if (pointOnLine !== undefined) {
                        const newGhostPoint = { idx: idx + 1, point: pointOnLine };
                        updateGhostPoint(newGhostPoint);
                    } else {
                        removeGhostPoint();
                    }
                };

                const onPointerDown = (event: PointerEvent) => {
                    const currentGhostPoint = ghostPointRef.current;
                    if (isLeftButton(event) && currentGhostPoint !== undefined) {
                        addPoint(currentGhostPoint.idx, currentGhostPoint.point.x, currentGhostPoint.point.y);
                    }
                };

                return (
                    <g onPointerLeave={removeGhostPoint} onPointerDown={onPointerDown} key={`${idx}`}>
                        <line
                            x1={point.x}
                            y1={point.y}
                            x2={nextPoint.x}
                            y2={nextPoint.y}
                            opacity={0}
                            stroke='black'
                            strokeWidth={`calc(${2 * ANCHOR_SIZE}px / var(--zoom-level))`}
                            onPointerMove={onPointerMove}
                            aria-label={`Line between point ${idx} and ${idx + 1}`}
                        />
                        {ghostPointRef.current === undefined ||
                        ghostPoint === undefined ||
                        ghostPoint.idx - 1 !== idx ? null : (
                            <ResizeAnchor
                                zoom={zoom}
                                cursor='default'
                                strokeWidth={0}
                                x={ghostPoint.point.x}
                                y={ghostPoint.point.y}
                                onComplete={onComplete}
                                fill={'var(--energy-blue)'}
                                label={`Add a point between point ${idx} and ${idx + 1}`}
                                moveAnchorTo={(x: number, y: number) => {
                                    updateGhostPoint({ ...ghostPoint, point: { x, y } });
                                    moveAnchorTo(ghostPoint.idx, x, y);
                                }}
                            />
                        )}
                    </g>
                );
            })}
        </>
    );
};
