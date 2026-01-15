// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, useEffect, useRef } from 'react';

import { getIntersectionPoint } from '@geti/smart-tools/utils';
import { isEmpty } from 'lodash-es';

import { SetStateWrapper } from '../../../dataset/media-preview/primary-toolbar/undo-redo/use-undo-redo-state';
import { Point, Polygon } from '../../types';
import {
    deleteSegments,
    ERASER_FIELD_DEFAULT_RADIUS,
    isCloseMode,
    leftRightMouseButtonHandler,
    MouseEventHandlers,
    PolygonMode,
    removeEmptySegments,
} from './utils';

interface UsePolygonProps {
    zoom: number;
    polygon: Polygon | null;
    lassoSegment: Point[];

    segments: Point[][];
    setSegments: SetStateWrapper<Point[][]>;

    mode: PolygonMode | null;
    setMode: (mode: PolygonMode | null) => void;

    canPathBeClosed: (point: Point) => boolean;
    setPointerLine: SetStateWrapper<Point[]>;
    setLassoSegment: SetStateWrapper<Point[]>;
    complete: (resetMode: PolygonMode | null) => void;
    setPointFromEvent: (callback: (point: Point) => void) => (event: PointerEvent<SVGElement>) => void;
    handleIsStartingPointHovered: (point: Point) => void;
}

export const usePolygon = ({
    zoom,
    mode,
    polygon,
    segments,
    lassoSegment,
    setMode,
    complete,
    setSegments,
    setPointerLine,
    canPathBeClosed,
    setLassoSegment,
    setPointFromEvent,
    handleIsStartingPointHovered,
}: UsePolygonProps): MouseEventHandlers => {
    const prevMainMode = useRef<PolygonMode | null>(null);
    const isPointerDown = useRef<boolean>(false);

    useEffect(() => {
        if (!isCloseMode(mode) && prevMainMode.current === PolygonMode.MagneticLasso) {
            setLassoSegment([]);
        }

        if (mode !== PolygonMode.Eraser && mode !== PolygonMode.Lasso) {
            prevMainMode.current = mode;
        }
    }, [mode, setLassoSegment]);

    const onPointerDown = leftRightMouseButtonHandler(
        (event) => {
            event.currentTarget.setPointerCapture(event.pointerId);

            setPointFromEvent((point: Point) => {
                setMode(PolygonMode.Polygon);

                isPointerDown.current = true;

                if (canPathBeClosed(point)) {
                    isPointerDown.current = false;

                    return;
                }

                setSegments(removeEmptySegments(lassoSegment, [point]));
                setLassoSegment([]);
            })(event);
        },
        (event) => {
            if (isEmpty(segments)) return;

            event.currentTarget.setPointerCapture(event.pointerId);

            setMode(PolygonMode.Eraser);
        }
    );

    const onPointerUp = (event: PointerEvent<SVGSVGElement>) => {
        event.currentTarget.releasePointerCapture(event.pointerId);

        setPointFromEvent((point: Point): void => {
            // finish the drawing while releasing the button inside the area of starting point
            if ((mode === PolygonMode.Lasso || isCloseMode(mode)) && polygon) {
                setSegments(removeEmptySegments(lassoSegment));
                setLassoSegment([]);
            }

            if (canPathBeClosed(point)) {
                //Note: to not clear snapping mode state
                const expectedMode = mode === PolygonMode.MagneticLassoClose ? mode : null;
                complete(expectedMode);
            }

            setMode(prevMainMode.current);
            isPointerDown.current = false;
        })(event);
    };

    const onPointerMove = setPointFromEvent((newPoint: Point) => {
        if (isEmpty(segments)) return;

        if (mode === PolygonMode.Polygon && isPointerDown.current) setMode(PolygonMode.Lasso);

        if (mode === PolygonMode.Lasso) {
            setLassoSegment((newLassoSegment: Point[]) => [...newLassoSegment, newPoint]);
        }

        if (mode === PolygonMode.Eraser) {
            const intersectionPoint = getIntersectionPoint(
                Math.ceil(ERASER_FIELD_DEFAULT_RADIUS / zoom),
                newPoint,
                segments.flat()
            );

            if (!intersectionPoint) return;

            setLassoSegment([]);
            setSegments(deleteSegments(intersectionPoint));
        }

        if (mode !== PolygonMode.Eraser) {
            handleIsStartingPointHovered(newPoint);
            setPointerLine(() => [...segments.flat(), ...lassoSegment, newPoint]);
        }
    });

    return { onPointerDown, onPointerUp, onPointerMove };
};
