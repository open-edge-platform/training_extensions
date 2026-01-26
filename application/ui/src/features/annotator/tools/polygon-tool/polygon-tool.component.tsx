// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, useRef, useState, useTransition } from 'react';

import { isPointOverPoint, isPolygonValid } from '@geti/smart-tools/utils';
import { isEmpty } from 'lodash-es';

import { useZoom } from '../../../../components/zoom/zoom.provider';
import { useAnnotationActions } from '../../../../shared/annotator/annotation-actions-provider.component';
import { useAnnotator } from '../../../../shared/annotator/annotator-provider.component';
import { Point } from '../../../../shared/types';
import { usePolygonConfig } from '../hooks/use-polygon-config.hook';
import { SvgToolCanvas } from '../svg-tool-canvas.component';
import { PolygonDraw } from './polygon-draw.component';
import {
    drawingStyles,
    getToolIcon,
    isCloseMode,
    isPolygonReadyToClose,
    leftRightMouseButtonHandler,
    PolygonMode,
    removeEmptySegments,
    START_POINT_FIELD_DEFAULT_RADIUS,
    START_POINT_FIELD_FOCUS_RADIUS,
} from './utils';

import classes from './polygon-tool.module.scss';

export const PolygonTool = () => {
    const { scale: zoom } = useZoom();
    const { addAnnotations } = useAnnotationActions();
    const { image, selectedLabel } = useAnnotator();

    const ref = useRef<SVGRectElement>({} as SVGRectElement);
    const isPointerDown = useRef<boolean>(false);
    const [mode, setMode] = useState<PolygonMode>(PolygonMode.Polygon);
    const [isPendingPolygonOptimization, startTransition] = useTransition();

    const {
        polygon,
        setPointerLine,
        lassoSegment,
        setLassoSegment,
        segments,
        setSegments,
        optimizePolygonOrSegments,
        resetTool,
        onPointerMoveRemove,
        setPointFromEvent,
    } = usePolygonConfig({ image, zoom, canvasRef: ref });

    const toolIcon = getToolIcon(mode);

    const STARTING_POINT_RADIUS = Math.ceil(
        (isCloseMode(mode) ? START_POINT_FIELD_FOCUS_RADIUS : START_POINT_FIELD_DEFAULT_RADIUS) / zoom
    );

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
            setMode(PolygonMode.PolygonClose);
        }

        if (isCloseMode(mode) && !canPathBeClosed(point)) {
            setMode(PolygonMode.Polygon);
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

            if (canPathBeClosed(point) && isPolygonReadyToClose(polygon)) {
                startTransition(async () => {
                    const optimizedPolygon = await optimizePolygonOrSegments(polygon);

                    addAnnotations(
                        [{ type: 'polygon', points: optimizedPolygon.points }],
                        selectedLabel ? [selectedLabel] : []
                    );
                });

                resetTool();
            }

            setMode(PolygonMode.Polygon);
            isPointerDown.current = false;
        })(event);
    };

    const onPointerMove = setPointFromEvent((newPoint: Point) => {
        if (isEmpty(segments)) return;

        if (mode === PolygonMode.Polygon && isPointerDown.current) setMode(PolygonMode.Lasso);

        if (mode === PolygonMode.Lasso) {
            setLassoSegment((newLassoSegment: Point[]) => [...newLassoSegment, newPoint]);
        }

        handleIsStartingPointHovered(newPoint);
        setPointerLine(() => [...segments.flat(), ...lassoSegment, newPoint]);
    });

    return (
        <SvgToolCanvas
            image={image}
            canvasRef={ref}
            aria-label={'polygon tool'}
            onPointerUp={onPointerUp}
            onPointerDown={onPointerDown}
            onPointerMove={mode === PolygonMode.Eraser ? onPointerMoveRemove : onPointerMove}
            onPointerLeave={mode === PolygonMode.Eraser ? onPointerMoveRemove : onPointerMove}
            style={{ cursor: `url(/icons/cursor/${toolIcon.icon}.png) ${toolIcon.offset}, auto` }}
        >
            {polygon !== null && !isEmpty(polygon.points) && (
                <PolygonDraw
                    shape={polygon}
                    ariaLabel='new polygon'
                    styles={drawingStyles(selectedLabel)}
                    className={isPendingPolygonOptimization ? classes.inputTool : ''}
                    indicatorRadius={STARTING_POINT_RADIUS}
                />
            )}
        </SvgToolCanvas>
    );
};
