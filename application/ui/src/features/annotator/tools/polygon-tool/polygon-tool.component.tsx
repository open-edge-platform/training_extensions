// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, useEffect, useRef, useState } from 'react';

import { clampPointBetweenImage, isPointOverPoint, isPolygonValid } from '@geti/smart-tools/utils';
import { isEmpty } from 'lodash-es';

import { useZoom } from '../../../../components/zoom/zoom.provider';
import { useAnnotationActions } from '../../../../shared/annotator/annotation-actions-provider.component';
import { useAnnotator } from '../../../../shared/annotator/annotator-provider.component';
import useUndoRedoState from '../../../dataset/media-preview/primary-toolbar/undo-redo/use-undo-redo-state';
import { Point, Polygon } from '../../types';
import { SvgToolCanvas } from '../svg-tool-canvas.component';
import { getRelativePoint } from '../utils';
import { PolygonDraw } from './polygon-draw.component';
import { usePolygon } from './use-polygon.hook';
import {
    drawingStyles,
    getCloseMode,
    getToolIcon,
    isCloseMode,
    PolygonMode,
    START_POINT_FIELD_DEFAULT_RADIUS,
    START_POINT_FIELD_FOCUS_RADIUS,
} from './utils';

import classes from './polygon-tool.module.scss';

export const PolygonTool = () => {
    const { scale: zoom } = useZoom();
    const { addAnnotations } = useAnnotationActions();
    const { image, selectedLabel } = useAnnotator();

    const ref = useRef<SVGRectElement>({} as SVGRectElement);
    const isClosing = useRef(false);
    const previousPolygonMode = useRef<PolygonMode | null>(null);

    const [mode, setMode] = useState<PolygonMode | null>(PolygonMode.Polygon);
    const [polygon, setPolygon] = useState<Polygon | null>(null);
    const [pointerLine, setPointerLine] = useState<Point[]>([]);
    const [lassoSegment, setLassoSegment] = useState<Point[]>([]);
    const [isOptimizingPolygons, setIsOptimizingPolygons] = useState(false);

    const [segments, setSegments, undoRedoActions] = useUndoRedoState<Point[][]>([]);

    const toolIcon = getToolIcon(mode);

    const STARTING_POINT_RADIUS = Math.ceil(
        (isCloseMode(mode) ? START_POINT_FIELD_FOCUS_RADIUS : START_POINT_FIELD_DEFAULT_RADIUS) / zoom
    );

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

    const complete = async () => {
        if (!polygon || isClosing.current) return;

        isClosing.current = true;
        polygon.points.pop();

        if (isPolygonValid({ shapeType: 'polygon', points: polygon.points })) {
            setIsOptimizingPolygons(true);

            onComplete(polygon);
            setIsOptimizingPolygons(false);
        }

        reset();
        isClosing.current = false;
    };

    useEffect((): void => setPolygon({ type: 'polygon', points: segments?.flat() }), [segments]);

    useEffect((): void => setPolygon({ type: 'polygon', points: pointerLine }), [pointerLine]);

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
            // store the previously used polygon mode
            previousPolygonMode.current = mode;

            setMode(getCloseMode(mode));
        }

        if (isCloseMode(mode) && !canPathBeClosed(point)) {
            setMode(previousPolygonMode.current);
        }
    };

    const polygonHandlers = usePolygon({
        mode,
        zoom,
        polygon,
        segments,
        lassoSegment,
        setMode,
        complete,
        setSegments,
        setPointerLine,
        setLassoSegment,
        canPathBeClosed,
        setPointFromEvent,
        handleIsStartingPointHovered,
    });

    const eventHandlers = polygonHandlers;

    return (
        <SvgToolCanvas
            image={image}
            canvasRef={ref}
            aria-label={'polygon tool'}
            onPointerUp={eventHandlers.onPointerUp}
            onPointerDown={eventHandlers.onPointerDown}
            onPointerMove={eventHandlers.onPointerMove}
            onPointerLeave={eventHandlers.onPointerMove}
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
