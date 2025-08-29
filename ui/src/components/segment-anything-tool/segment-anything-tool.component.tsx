// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, useEffect, useRef, useState } from 'react';

import { clampPointBetweenImage, isPointInShape, pointInRectangle } from '@geti/smart-tools/utils';

import { isRightButton } from '../annotation/utils';
import { Annotation, Point, RegionOfInterest, Shape } from '../shapes/interfaces';
import { getRelativePoint, removeOffLimitPoints } from '../shapes/utils';
import { useZoom } from '../zoom/zoom';
import { InteractiveSegmentationPoint } from './interactive-segmentation-point.component';
import { useSegmentAnything } from './segment-anything-state-provider.component';
import { InteractiveAnnotationPoint } from './segment-anything.interface';
import { useSingleStackFn } from './use-single-stack-fn.hook';
import { useThrottledCallback } from './use-throttled-callback.hook';

import classes from './segment-anything.module.scss';

const isPositivePoint = (point: Point, shapes: Shape[], isRightClick: boolean, rightClickMode: boolean) => {
    if (rightClickMode) {
        return !isRightClick;
    }

    return !shapes.some((shape) => isPointInShape(shape, point));
};

// Whenever the user moves their mouse over the canvas  we compute a preview of
// SAM being applied to the user's mouse position.
// The decoding step of SAM takes on average 100ms with 150-250ms being a high
// exception. We throttle the mouse update based on this so that we don't overload
// the user's cpu with too many decoding requests
const THROTTLE_TIME = 150;

const selectedMediaItem = {
    identifier: 'id',
    image: new ImageData(100, 100),
};

const roi: RegionOfInterest = {
    x: 0,
    y: 0,
    width: 100,
    height: 100,
};

const toolSettings = {
    interactiveMode: false,
    rightClickMode: false,
};

export const SegmentAnythingTool = () => {
    const zoom = useZoom();

    const clampPoint = clampPointBetweenImage(selectedMediaItem.image);

    const ref = useRef<SVGRectElement>(null);

    const { result, points, addPoint } = useSegmentAnything();
    const [mousePosition, setMousePosition] = useState<InteractiveAnnotationPoint>();

    const [previewShapes, setPreviewShapes] = useState<Shape[]>([]);
    const { decodingQueryFn } = useSegmentAnything();
    const throttledDecodingQueryFn = useSingleStackFn(decodingQueryFn);

    const throttleSetMousePosition = useThrottledCallback((point: InteractiveAnnotationPoint) => {
        setMousePosition(point);
    }, THROTTLE_TIME);

    useEffect(() => {
        if (mousePosition === undefined) {
            return;
        }

        throttledDecodingQueryFn([mousePosition])
            .then((shapes) => {
                setPreviewShapes(shapes.map((shape) => removeOffLimitPoints(shape, roi)));

                throttleSetMousePosition.flush();
            })
            .catch(() => {
                // If getting decoding went wrong we set an empty preview and
                // start to compute the next decoding
                return [];
            });
    }, [mousePosition, throttledDecodingQueryFn, throttleSetMousePosition]);

    const { interactiveMode, rightClickMode } = toolSettings;

    const handleMouseMove = (event: PointerEvent<SVGSVGElement>) => {
        if (!ref.current) {
            return;
        }

        const point = clampPoint(getRelativePoint(ref.current, { x: event.clientX, y: event.clientY }, zoom.scale));

        // In task chain don't allow the user to place a point outside the ROI
        if (!pointInRectangle(roi, point)) {
            return;
        }

        const positive = isPositivePoint(point, result.shapes, isRightButton(event), rightClickMode);

        throttleSetMousePosition({ ...point, positive });
    };

    const onPointerUp = (event: PointerEvent<SVGSVGElement>) => {
        if (!ref.current) {
            return;
        }

        if (event.button !== 0 && event.button !== 2) {
            return;
        }

        if (event.pointerType === 'touch') {
            return;
        }

        if (!rightClickMode && isRightButton(event)) {
            return;
        }

        // The user must first place a positive point as otherwise we can't show a preview
        if (rightClickMode && isRightButton(event) && points.length === 0) {
            return;
        }

        const point = clampPoint(getRelativePoint(ref.current, { x: event.clientX, y: event.clientY }, zoom));

        // In task chain don't allow the user to place a point outside the ROI
        if (!pointInRectangle(roi, point)) {
            return;
        }

        const positive = isPositivePoint(point, result.shapes, isRightButton(event), rightClickMode);

        const shouldKeepPreviousPoints = interactiveMode === true || isRightButton(event);

        addPoint({ x: point.x, y: point.y, positive }, shouldKeepPreviousPoints);
    };

    const showPreviewShapes = result.shapes.length === 0 && mousePosition !== undefined;
    const annotations = (showPreviewShapes ? previewShapes : result.shapes).map((shape, idx): Annotation => {
        return {
            shape,
            labels: [],
            id: `${idx}`,
            isHidden: false,
            isLocked: false,
            isSelected: false,
            zIndex: idx,
        };
    });

    return (
        <SvgToolCanvas
            image={image}
            canvasRef={ref}
            onPointerMove={handleMouseMove}
            onPointerUp={onPointerUp}
            onPointerLeave={() => {
                throttleSetMousePosition.cancel();
                setMousePosition(undefined);
                setPreviewShapes([]);
            }}
            style={{
                cursor: interactiveMode
                    ? `url("/icons/pencil-plus.svg") 16 16, auto`
                    : `url("/icons/selection.svg") 8 8, auto`,
            }}
        >
            <AnnotationsMask
                fillOpacity={toolSettings.maskOpacity}
                annotations={annotations}
                width={image.width}
                height={image.height}
            />

            {showPreviewShapes &&
                previewShapes.map((shape, idx) => (
                    <g
                        key={idx}
                        aria-label='Segment anything preview'
                        {...SELECT_ANNOTATION_STYLES}
                        strokeWidth={'calc(3px / var(--zoom-level))'}
                        fillOpacity={0.0}
                        className={interactiveMode ? classes.stroke : classes.animateStroke}
                    >
                        <ShapeFactory annotation={{ shape, id: '', isSelected: false }} />
                    </g>
                ))}

            {result.shapes.map((shape, idx) => (
                <g
                    key={idx}
                    aria-label='Segment anything result'
                    {...SELECT_ANNOTATION_STYLES}
                    strokeWidth={'calc(3px / var(--zoom-level))'}
                    cursor={`url(/icons/cursor/pencil-${
                        interactiveMode === true && rightClickMode === false ? 'minus' : 'plus'
                    }.svg) 16 16, auto`}
                    fillOpacity={0.0}
                    className={classes.stroke}
                >
                    <ShapeFactory annotation={{ shape, id: '', isSelected: false }} />
                </g>
            ))}

            {points.map((point, index) => (
                <InteractiveSegmentationPoint
                    key={index}
                    x={point.x}
                    y={point.y}
                    positive={point.positive}
                    isLoading={false}
                />
            ))}
        </SvgToolCanvas>
    );
};
