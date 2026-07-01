// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, RefObject, useEffect, useMemo, useRef } from 'react';

import { clampPointBetweenImage, getIntersectionPoint } from '@geti/smart-tools/utils';
import { differenceWith, isEmpty, isEqual, isNil } from 'lodash-es';

import { Point, Polygon } from '../../../../shared/types';
import { usePolygonState } from '../polygon-tool/polygon-state-provider.component';
import { deleteSegments, ERASER_FIELD_DEFAULT_RADIUS } from '../polygon-tool/utils';
import { convertToolShapeToGetiShape, getRelativePoint } from '../utils';
import { useIntelligentScissorsWorker } from './use-intelligent-scissors-worker.hook';

export const usePolygonConfig = ({
    zoom,
    image,
    canvasRef,
}: {
    zoom: number;
    image: ImageData;
    canvasRef: RefObject<SVGRectElement>;
}) => {
    const isMounted = useRef(true);
    const { worker } = useIntelligentScissorsWorker();

    const { segments, setSegments, pointerLine, setPointerLine, lassoSegment, setLassoSegment, undoRedoActions } =
        usePolygonState();

    useEffect(() => {
        isMounted.current = true;

        return () => {
            isMounted.current = false;
            worker?.cleanImg();
        };
    }, [worker]);

    useEffect(() => {
        if (isMounted.current && image) {
            worker?.loadImage(image);
        }
    }, [image, worker]);

    const polygon = useMemo<Polygon | null>(() => {
        const points = pointerLine.length > 0 ? pointerLine : segments.flat();

        return points.length > 0 ? { type: 'polygon', points } : null;
    }, [pointerLine, segments]);

    const optimizePolygonOrSegments = async (iPolygon: Polygon): Promise<Polygon> => {
        if (isNil(worker)) {
            return Promise.reject(new Error('Intelligent scissors worker not initialized'));
        }

        const lastSegment = differenceWith(iPolygon.points, segments.flat(), isEqual);
        const newSegments = isEmpty(lastSegment) ? [...segments] : [...segments, lastSegment];

        const resultPolygon = await worker.optimizeSegments(newSegments);
        return convertToolShapeToGetiShape(resultPolygon);
    };

    const getPointerRelativePosition = (event: PointerEvent<SVGElement>): Point => {
        const clampPoint = clampPointBetweenImage(image);

        return clampPoint(getRelativePoint(canvasRef.current, { x: event.clientX, y: event.clientY }, zoom));
    };

    const setPointFromEvent = (callback: (point: Point) => void) => (event: PointerEvent<SVGElement>) =>
        callback(getPointerRelativePosition(event));

    const onPointerMoveRemove = setPointFromEvent((newPoint: Point) => {
        const intersectionPoint = getIntersectionPoint(
            Math.ceil(ERASER_FIELD_DEFAULT_RADIUS / zoom),
            newPoint,
            segments.flat()
        );

        if (!intersectionPoint) return;

        setLassoSegment([]);
        setSegments(deleteSegments(intersectionPoint));
    });

    const resetTool = (): void => {
        undoRedoActions.reset();

        setPointerLine([]);
        setLassoSegment([]);
    };

    return {
        worker,
        polygon,
        setPointerLine,
        lassoSegment,
        setLassoSegment,
        isMounted: isMounted.current,
        segments,
        setSegments,
        optimizePolygonOrSegments,
        resetTool,
        onPointerMoveRemove,
        setPointFromEvent,
    };
};
