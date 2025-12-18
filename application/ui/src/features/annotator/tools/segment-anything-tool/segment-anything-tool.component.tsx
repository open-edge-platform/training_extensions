// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, useEffect, useRef, useState } from 'react';

import { clampPointBetweenImage } from '@geti/smart-tools/utils';

import { useZoom } from '../../../../components/zoom/zoom.provider';
import { useAnnotator } from '../../../../shared/annotator/annotator-provider.component';
import { AnnotationShapeWithLabels } from '../../annotations/annotation-shape-with-labels.component';
import { MaskAnnotations } from '../../annotations/mask-annotations.component';
import type { Annotation, Shape } from '../../types';
import { SvgToolCanvas } from '../svg-tool-canvas.component';
import { getRelativePoint, removeOffLimitPoints } from '../utils';
import { SAMLoading } from './sam-loading.component';
import { InteractiveAnnotationPoint } from './segment-anything.interface';
import { useDecodingMutation } from './use-decoding-query.hook';
import { useSegmentAnythingModel } from './use-segment-anything.hook';
import { useSingleStackFn } from './use-single-stack-fn.hook';
import { useThrottledCallback } from './use-throttle-callback.hook';

import classes from './segment-anything.module.scss';

// Whenever the user moves their mouse over the canvas  we compute a preview of
// SAM being applied to the user's mouse position.
// The decoding step of SAM takes on average 100ms with 150-250ms being a high
// exception. We throttle the mouse update based on this so that we don't overload
// the user's cpu with too many decoding requests
const THROTTLE_TIME = 150;

const SELECT_ANNOTATION_STYLES = {
    fillOpacity: 0.3,
    fill: 'var(--energy-blue-shade)',
    stroke: 'var(--energy-blue-shade)',
    strokeWidth: 'calc(2px / var(--zoom-scale))',
};

export const SegmentAnythingTool = () => {
    const [mousePosition, setMousePosition] = useState<InteractiveAnnotationPoint>();
    const [previewShapes, setPreviewShapes] = useState<Shape[]>([]);

    const zoom = useZoom();
    const { mediaItem, roi, image } = useAnnotator();
    const { isLoading, decodingQueryFn } = useSegmentAnythingModel();
    const throttledDecodingQueryFn = useSingleStackFn(decodingQueryFn);
    const decodingMutation = useDecodingMutation(decodingQueryFn);

    const ref = useRef<SVGRectElement>(null);

    const clampPoint = clampPointBetweenImage(image);

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

    const handleMouseMove = (event: PointerEvent<SVGSVGElement>) => {
        if (!ref.current) {
            return;
        }

        const point = clampPoint(getRelativePoint(ref.current, { x: event.clientX, y: event.clientY }, zoom.scale));

        throttleSetMousePosition({ ...point, positive: true });
    };

    const onPointerUp = (event: PointerEvent<SVGSVGElement>) => {
        if (!ref.current) {
            return;
        }

        if (event.button !== 0 && event.button !== 2) {
            return;
        }

        const point = clampPoint(getRelativePoint(ref.current, { x: event.clientX, y: event.clientY }, zoom.scale));

        decodingMutation.mutate([{ ...point, positive: true }]);
    };

    const annotations = previewShapes.map((shape, idx): Annotation => {
        return {
            shape,
            labels: [{ id: 'id', color: 'red', name: 'Segment Anything', isPrediction: false }],
            id: `${idx}`,
        };
    });

    if (isLoading) {
        return <SAMLoading isLoading={isLoading} />;
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
                cursor: `url("/icons/selection.svg") 8 8, auto`,
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
                        <AnnotationShapeWithLabels
                            annotation={{
                                shape,
                                id: '',
                                labels: [{ id: 'id', color: 'red', name: 'Segment Anything', isPrediction: false }],
                            }}
                        />
                    </g>
                ))}
        </SvgToolCanvas>
    );
};
