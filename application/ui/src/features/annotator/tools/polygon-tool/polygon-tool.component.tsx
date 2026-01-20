// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, useEffect, useRef, useState } from 'react';

import {
    clampPointBetweenImage,
    getIntersectionPoint,
    isPointOverPoint,
    isPolygonValid,
} from '@geti/smart-tools/utils';
import { differenceWith, isEmpty, isEqual, isNil } from 'lodash-es';

import { useZoom } from '../../../../components/zoom/zoom.provider';
import { useAnnotationActions } from '../../../../shared/annotator/annotation-actions-provider.component';
import { useAnnotator } from '../../../../shared/annotator/annotator-provider.component';
import { Point, Polygon } from '../../../../shared/types';
import useUndoRedoState from '../../../dataset/media-preview/primary-toolbar/undo-redo/use-undo-redo-state';
import { SvgToolCanvas } from '../svg-tool-canvas.component';
import { convertToolShapeToGetiShape, getRelativePoint } from '../utils';
import { PolygonDraw } from './polygon-draw.component';
import { useIntelligentScissorsWorker } from './use-intelligent-scissors-worker.hook';
import {
    deleteSegments,
    drawingStyles,
    ERASER_FIELD_DEFAULT_RADIUS,
    getCloseMode,
    getToolIcon,
    isCloseMode,
    leftRightMouseButtonHandler,
    PolygonMode,
    removeEmptySegments,
    START_POINT_FIELD_DEFAULT_RADIUS,
    START_POINT_FIELD_FOCUS_RADIUS,
} from './utils';

import classes from './polygon-tool.module.scss';

interface PolygonToolProps {
    mainMode: PolygonMode;
}

export const PolygonTool = ({ mainMode }: PolygonToolProps) => {
    const { scale: zoom } = useZoom();
    const { addAnnotations } = useAnnotationActions();
    const { image, selectedLabel } = useAnnotator();

    const ref = useRef<SVGRectElement>({} as SVGRectElement);
    const isClosing = useRef(false);
    const isPointerDown = useRef<boolean>(false);

    const [mode, setMode] = useState<PolygonMode | null>(mainMode);
    const [polygon, setPolygon] = useState<Polygon | null>(null);
    const [pointerLine, setPointerLine] = useState<Point[]>([]);
    const [lassoSegment, setLassoSegment] = useState<Point[]>([]);
    const [isOptimizingPolygons, setIsOptimizingPolygons] = useState(false);
    const { worker } = useIntelligentScissorsWorker();

    const [segments, setSegments, undoRedoActions] = useUndoRedoState<Point[][]>([]);

    const toolIcon = getToolIcon(mode);

    const STARTING_POINT_RADIUS = Math.ceil(
        (isCloseMode(mode) ? START_POINT_FIELD_FOCUS_RADIUS : START_POINT_FIELD_DEFAULT_RADIUS) / zoom
    );

    useEffect((): void => setPolygon({ type: 'polygon', points: segments?.flat() }), [segments]);
    useEffect((): void => setPolygon({ type: 'polygon', points: pointerLine }), [pointerLine]);

    const reset = (): void => {
        undoRedoActions.reset();

        setPointerLine([]);
        setLassoSegment([]);
        setPolygon(null);
    };

    const onComplete = ({ points }: Polygon) => {
        !isEmpty(points) && addAnnotations([{ type: 'polygon', points }], selectedLabel ? [selectedLabel] : []);
    };

    const getPointerRelativePosition = (event: PointerEvent<SVGElement>): Point => {
        const clampPoint = clampPointBetweenImage(image);

        return clampPoint(getRelativePoint(ref.current, { x: event.clientX, y: event.clientY }, zoom));
    };

    const optimizePolygonOrSegments = async (iPolygon: Polygon): Promise<Polygon> => {
        if (isNil(worker)) {
            return Promise.reject(new Error('Intelligent scissors worker not initialized'));
        }

        const lastSegment = differenceWith(iPolygon.points, segments.flat(), isEqual);
        const newSegments = isEmpty(lastSegment) ? [...segments] : [...segments, lastSegment];

        const resultPolygon = await worker.optimizeSegments(newSegments);
        return convertToolShapeToGetiShape(resultPolygon);
    };

    const complete = async () => {
        if (!polygon || isClosing.current) return;

        isClosing.current = true;
        polygon.points.pop();

        if (isPolygonValid({ shapeType: 'polygon', points: polygon.points })) {
            setIsOptimizingPolygons(true);

            const optimizedPolygon = await optimizePolygonOrSegments(polygon);
            onComplete(optimizedPolygon);
            setIsOptimizingPolygons(false);
        }

        reset();
        isClosing.current = false;
    };

    const setPointFromEvent = (callback: (point: Point) => void) => (event: PointerEvent<SVGElement>) =>
        callback(getPointerRelativePosition(event));

    const canPathBeClosed = (point: Point): boolean => {
        const flatSegments = segments.flat();

        if (isEmpty(flatSegments)) return false;

        return (
            Boolean(polygon) &&
            isPolygonValid({ shapeType: 'polygon', points: polygon?.points ?? [] }) &&
            isPointOverPoint(flatSegments[0], point, STARTING_POINT_RADIUS)
        );
    };

    const handleIsStartingPointHovered = (point: Point): void => {
        if (!isCloseMode(mode) && canPathBeClosed(point)) {
            setMode(getCloseMode(mode));
        }

        if (isCloseMode(mode) && !canPathBeClosed(point)) {
            setMode(mainMode);
        }
    };

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
                complete();
            }

            setMode(mainMode);
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

    return (
        <SvgToolCanvas
            image={image}
            canvasRef={ref}
            aria-label={'polygon tool'}
            onPointerUp={onPointerUp}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerLeave={onPointerMove}
            style={{ cursor: `url(/icons/cursor/${toolIcon.icon}.png) ${toolIcon.offset}, auto` }}
        >
            {polygon !== null && !isEmpty(polygon.points) && (
                <PolygonDraw
                    shape={polygon}
                    ariaLabel='new polygon'
                    styles={drawingStyles(selectedLabel)}
                    className={isOptimizingPolygons ? classes.inputTool : ''}
                    indicatorRadius={STARTING_POINT_RADIUS}
                />
            )}
        </SvgToolCanvas>
    );
};
