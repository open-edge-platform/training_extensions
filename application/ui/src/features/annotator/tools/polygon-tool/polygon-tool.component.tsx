// Copyright (C) 2022-2025 Intel Corporation
// LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import { PointerEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { Polygon as GetiPolygon } from '@geti/smart-tools/types';
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
    isCloseMode,
    PointerIcons,
    PointerIconsOffset,
    PolygonMode,
    START_POINT_FIELD_DEFAULT_RADIUS,
    START_POINT_FIELD_FOCUS_RADIUS,
} from './utils';

import classes from './polygon-tool.module.scss';

const getCloseMode = (mode: PolygonMode | null) => {
    if (mode === PolygonMode.MagneticLasso) {
        return PolygonMode.MagneticLassoClose;
    }

    if (mode === PolygonMode.Lasso) {
        return PolygonMode.LassoClose;
    }

    return PolygonMode.PolygonClose;
};

const TOOL_ICON: Record<PolygonMode, { icon: PointerIcons; offset: PointerIconsOffset }> = {
    [PolygonMode.Polygon]: { icon: PointerIcons.Polygon, offset: PointerIconsOffset.Polygon },
    [PolygonMode.Eraser]: { icon: PointerIcons.Eraser, offset: PointerIconsOffset.Eraser },
    [PolygonMode.Lasso]: { icon: PointerIcons.Lasso, offset: PointerIconsOffset.Lasso },
    [PolygonMode.LassoClose]: { icon: PointerIcons.LassoClose, offset: PointerIconsOffset.LassoClose },
    [PolygonMode.MagneticLasso]: { icon: PointerIcons.MagneticLasso, offset: PointerIconsOffset.MagneticLasso },
    [PolygonMode.MagneticLassoClose]: {
        icon: PointerIcons.MagneticLassoClose,
        offset: PointerIconsOffset.MagneticLassoClose,
    },
    [PolygonMode.PolygonClose]: { icon: PointerIcons.PolygonClose, offset: PointerIconsOffset.PolygonClose },
};

export const PolygonTool = () => {
    const { scale: zoom } = useZoom();
    const { addAnnotations } = useAnnotationActions();
    const { mediaItem, image, selectedLabel } = useAnnotator();

    const ref = useRef<SVGRectElement>({} as SVGRectElement);
    const isClosing = useRef(false);
    const previousPolygonMode = useRef<PolygonMode | null>(null);

    const [polygon, setPolygon] = useState<Polygon | null>(null);
    const [pointerLine, setPointerLine] = useState<Point[]>([]);
    const [lassoSegment, setLassoSegment] = useState<Point[]>([]);
    const [isOptimizingPolygons, setIsOptimizingPolygons] = useState(false);

    const [segments, setSegments, undoRedoActions] = useUndoRedoState<Point[][]>([]);
    const [mode, setMode] = useState<PolygonMode | null>(PolygonMode.Polygon);

    // START_POINT_FIELD_FOCUS_RADIUS / zoom ~ 16 px
    // START_POINT_FIELD_DEFAULT_RADIUS / zoom ~ 12 px
    const STARTING_POINT_RADIUS = useMemo(
        () => Math.ceil((isCloseMode(mode) ? START_POINT_FIELD_FOCUS_RADIUS : START_POINT_FIELD_DEFAULT_RADIUS) / zoom),
        [mode, zoom]
    );

    const toolIcon = TOOL_ICON[PolygonMode.Polygon];

    const reset = useCallback(
        (): void => {
            undoRedoActions.reset();

            setPointerLine([]);
            setLassoSegment([]);
            setPolygon(null);
        },
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [undoRedoActions.reset]
    );

    const onComplete = (optimizedPolygon: Polygon) => {
        !isEmpty(optimizedPolygon.points) &&
            addAnnotations(
                [{ type: 'polygon', points: optimizedPolygon.points }],
                selectedLabel ? [selectedLabel] : []
            );
    };

    const getPointerRelativePosition = useCallback(
        (event: PointerEvent<SVGElement>): Point => {
            const clampPoint = clampPointBetweenImage(image);

            return clampPoint(getRelativePoint(ref.current, { x: event.clientX, y: event.clientY }, zoom));
        },
        [image, zoom]
    );

    const complete = useCallback(
        async (inputMode: PolygonMode | null) => {
            if (!polygon || isClosing.current) return;

            isClosing.current = true;
            polygon.points.pop();

            if (isPolygonValid(polygon as unknown as GetiPolygon)) {
                setIsOptimizingPolygons(true);

                onComplete(polygon);
                setIsOptimizingPolygons(false);
            }

            reset();
            //resetMode(inputMode);
            isClosing.current = false;
        },
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [onComplete, polygon, reset, mode]
    );

    useEffect((): void => setPolygon({ type: 'polygon', points: segments?.flat() }), [segments]);

    useEffect((): void => setPolygon({ type: 'polygon', points: pointerLine }), [pointerLine]);

    const setPointFromEvent = (callback: (point: Point) => void) => (event: PointerEvent<SVGElement>) =>
        callback(getPointerRelativePosition(event));

    const canPathBeClosed = useCallback(
        (point: Point): boolean => {
            const flatSegments = segments.flat();

            if (isEmpty(flatSegments)) return false;

            return (
                Boolean(polygon) &&
                isPolygonValid(polygon as unknown as GetiPolygon) &&
                isPointOverPoint(flatSegments[0], point, STARTING_POINT_RADIUS)
            );
        },
        [polygon, segments, STARTING_POINT_RADIUS]
    );

    const handleIsStartingPointHovered = useCallback(
        (point: Point): void => {
            if (!isCloseMode(mode) && canPathBeClosed(point)) {
                // store the previously used polygon mode
                previousPolygonMode.current = mode;

                setMode(getCloseMode(mode));
            }

            if (isCloseMode(mode) && !canPathBeClosed(point)) {
                setMode(previousPolygonMode.current);
            }
        },
        [mode, canPathBeClosed, previousPolygonMode, setMode]
    );

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

    console.log('toolIcon', toolIcon);
    return (
        <SvgToolCanvas
            image={image}
            canvasRef={ref}
            onPointerUp={(event) => {
                eventHandlers.onPointerUp(event);
            }}
            onPointerDown={(event) => {
                eventHandlers.onPointerDown(event);
            }}
            onPointerMove={eventHandlers.onPointerMove}
            onPointerLeave={eventHandlers.onPointerMove}
            style={{ cursor: `url(/icons/cursor/${toolIcon.icon}.png) ${toolIcon.offset}, auto` }}
            aria-label='polygon tool'
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
