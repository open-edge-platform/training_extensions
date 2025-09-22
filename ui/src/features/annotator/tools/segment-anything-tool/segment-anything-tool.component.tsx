// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, useEffect, useRef, useState } from 'react';

import { clampPointBetweenImage, isPointInShape, pointInRectangle } from '@geti/smart-tools/utils';

import { useZoom } from '../../../../components/zoom/zoom';
import { AnnotationShape } from '../../annotations/annotation-shape.component';
import { MaskAnnotations } from '../../annotations/mask-annotations.component';
import { useAnnotator } from '../../annotator-provider.component';
import { AnnotatorLoading } from '../../loading.component';
import { Annotation, Point, Shape } from '../../types';
import { isRightButton } from '../../utils';
import { SvgToolCanvas } from '../svg-tool-canvas.component';
import { getRelativePoint, removeOffLimitPoints } from '../utils';
import { InteractiveSegmentationPoint } from './interactive-segmentation-point.component';
import { useSegmentAnything } from './segment-anything-state-provider.component';
import { InteractiveAnnotationPoint } from './segment-anything.interface';
import { useSingleStackFn } from './use-single-stack-fn.hook';
import { useThrottledCallback } from './use-throttle-callback.hook';

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

// TODO: Remove/move this to the secondary toolbar
const toolSettings = {
    interactiveMode: false,
    rightClickMode: false,
};

const SELECT_ANNOTATION_STYLES = {
    fillOpacity: 0.3,
    fill: 'var(--energy-blue-shade)',
    stroke: 'var(--energy-blue-shade)',
    strokeWidth: 'calc(2px / var(--zoom-scale))',
};

export const SegmentAnythingTool = () => {
    const zoom = useZoom();
    const { mediaItem, roi, image } = useAnnotator();

    const clampPoint = clampPointBetweenImage(image);

    const ref = useRef<SVGRectElement>(null);

    const { result, points, addPoint, isProcessing, isLoading } = useSegmentAnything();
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
    }, [mousePosition, throttledDecodingQueryFn, throttleSetMousePosition, roi]);

    const { interactiveMode, rightClickMode } = toolSettings;

    const handleMouseMove = (event: PointerEvent<SVGSVGElement>) => {
        if (!ref.current) {
            return;
        }

        const point = clampPoint(getRelativePoint(ref.current, { x: event.clientX, y: event.clientY }, zoom.scale));

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

        const point = clampPoint(getRelativePoint(ref.current, { x: event.clientX, y: event.clientY }, zoom.scale));

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
            labels: [{ id: 'id', color: 'red', name: 'Segment Anything', isPrediction: false }],
            id: `${idx}`,
        };
    });

    if (isLoading) {
        return <AnnotatorLoading isLoading={isLoading} />;
    }

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
            <MaskAnnotations isEnabled annotations={annotations} width={mediaItem.width} height={mediaItem.height}>
                <></>
            </MaskAnnotations>

            {previewShapes.length > 0 &&
                previewShapes.map((shape, idx) => (
                    <g
                        key={idx}
                        aria-label='Segment anything preview'
                        {...SELECT_ANNOTATION_STYLES}
                        strokeWidth={'calc(3px / var(--zoom-scale))'}
                        fillOpacity={0.0}
                        className={classes.stroke}
                    >
                        <AnnotationShape
                            annotation={{
                                shape,
                                id: '',
                                labels: [{ id: 'id', color: 'red', name: 'Segment Anything', isPrediction: false }],
                            }}
                        />
                    </g>
                ))}

            {result.shapes.map((shape, idx) => (
                <g
                    key={idx}
                    aria-label='Segment anything result'
                    {...SELECT_ANNOTATION_STYLES}
                    strokeWidth={'calc(3px / var(--zoom-scale))'}
                    cursor={`url(/icons/pencil-${
                        interactiveMode === true && rightClickMode === false ? 'minus' : 'plus'
                    }.svg) 16 16, auto`}
                    fillOpacity={0.0}
                    className={classes.stroke}
                >
                    <AnnotationShape
                        annotation={{
                            shape,
                            id: '',
                            labels: [{ id: 'id', color: 'red', name: 'Segment Anything', isPrediction: false }],
                        }}
                    />
                </g>
            ))}

            {points.map((point, index) => (
                <InteractiveSegmentationPoint
                    key={index}
                    x={point.x}
                    y={point.y}
                    positive={point.positive}
                    isLoading={isProcessing}
                />
            ))}
        </SvgToolCanvas>
    );
};
