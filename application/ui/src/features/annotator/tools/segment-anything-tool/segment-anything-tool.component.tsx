// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, useCallback, useRef, useState } from 'react';

import { clampPointBetweenImage } from '@geti/smart-tools/utils';

import selectionCursor from '../../../../assets/icons/selection.svg?url';
import { useZoom } from '../../../../components/zoom/zoom.provider';
import type { Label } from '../../../../constants/shared-types';
import { useAnnotator } from '../../../../shared/annotator/annotator-provider.component';
import type { Annotation, RegionOfInterest, Shape } from '../../../../shared/types';
import { AnnotationShape } from '../../annotations/annotation-shape/annotation-shape.component';
import { MaskAnnotations } from '../../annotations/mask-annotations.component';
import { SvgToolCanvas } from '../svg-tool-canvas.component';
import { useAddAndSelectAnnotations } from '../use-add-and-select-annotations.hook';
import { getRelativePoint, removeOffLimitPoints } from '../utils';
import { SAMLoading } from './sam-loading.component';
import { InteractiveAnnotationPoint } from './segment-anything.interface';
import { useSegmentAnythingModel } from './use-segment-anything.hook';
import { useSingleStackFn } from './use-single-stack-fn.hook';

import classes from './segment-anything.module.scss';

interface PreviewAnnotationsProps {
    previewAnnotations: Annotation[];
    image: Pick<RegionOfInterest, 'width' | 'height'>;
}

const CURSOR_OFFSET = '7 8';

const PreviewAnnotations = ({ previewAnnotations, image }: PreviewAnnotationsProps) => {
    if (previewAnnotations.length === 0) return null;

    return (
        <MaskAnnotations isEnabled annotations={previewAnnotations} width={image.width} height={image.height}>
            {previewAnnotations.map((annotation) => (
                <g
                    key={annotation.id}
                    aria-label='Segment anything preview'
                    stroke={'var(--energy-blue-shade)'}
                    strokeWidth={'calc(3px / var(--zoom-scale))'}
                    fill={'transparent'}
                    fillOpacity={'var(--annotation-fill-opacity)'}
                    className={classes.animateStroke}
                >
                    <AnnotationShape annotation={annotation} />
                </g>
            ))}
        </MaskAnnotations>
    );
};

const useWithCancel = (fn: (points: InteractiveAnnotationPoint[]) => Promise<Shape[]>) => {
    const abortController = useRef(new AbortController());

    const cancellableCallback = useCallback(
        async (...args: Parameters<typeof fn>) => {
            abortController.current = new AbortController();

            const result = await fn(...args);

            if (abortController.current.signal.aborted) {
                throw new Error('Aborted');
            }

            return result;
        },
        [fn]
    );

    return {
        call: cancellableCallback,
        cancel: () => abortController.current.abort(),
    };
};

export const SegmentAnythingTool = () => {
    const [previewShapes, setPreviewShapes] = useState<Shape[]>([]);
    const [acceptedShapes, setAcceptedShapes] = useState<Shape[] | null>(null);
    const ref = useRef<SVGSVGElement>(null);

    const zoom = useZoom();
    const { roi, image, selectedLabel } = useAnnotator();
    const { addAndSelectAnnotations } = useAddAndSelectAnnotations();
    const { isLoading, decodingQueryFn } = useSegmentAnythingModel();
    const throttledDecodingQueryFn = useSingleStackFn(decodingQueryFn);
    const cancellableThrottledDecodingQueryFn = useWithCancel(throttledDecodingQueryFn);

    const canvasRef = useRef<SVGRectElement>(null);

    const clampPoint = clampPointBetweenImage(image);

    const handleMouseMove = (event: PointerEvent<SVGSVGElement>) => {
        if (acceptedShapes !== null) {
            return;
        }

        if (!canvasRef.current) {
            return;
        }

        const point = clampPoint(
            getRelativePoint(canvasRef.current, { x: event.clientX, y: event.clientY }, zoom.scale)
        );

        cancellableThrottledDecodingQueryFn
            .call([{ ...point, positive: true }])
            .then((shapes) => {
                setPreviewShapes(shapes.map((shape) => removeOffLimitPoints(shape, roi)));
            })
            .catch(() => {
                // If getting decoding went wrong we set an empty preview and
                // start to compute the next decoding
                return [];
            });
    };

    const handleAddAnnotations = (shapes: Shape[], label: Label) => {
        addAndSelectAnnotations(shapes, [label]);
        setPreviewShapes([]);
    };

    const handlePointerDown = (event: PointerEvent<SVGSVGElement>) => {
        if (!ref.current) {
            return;
        }

        if (event.button !== 0 && event.button !== 2) {
            return;
        }

        if (previewShapes.length === 0) {
            return;
        }

        if (selectedLabel == null) {
            setAcceptedShapes(previewShapes);

            return;
        }

        handleAddAnnotations(previewShapes, selectedLabel);
    };

    const handlePointerLeave = () => {
        cancellableThrottledDecodingQueryFn.cancel();
        setPreviewShapes([]);
    };

    const previewAnnotations = (acceptedShapes ?? previewShapes).map((shape, idx): Annotation => {
        return {
            shape,
            labels: selectedLabel ? [selectedLabel] : [],
            id: `${idx}`,
        };
    });

    if (isLoading) {
        return <SAMLoading isLoading={isLoading} />;
    }

    return (
        <SvgToolCanvas
            ref={ref}
            image={image}
            canvasRef={canvasRef}
            aria-label='SAM tool canvas'
            onPointerMove={handleMouseMove}
            onPointerDown={handlePointerDown}
            onPointerLeave={handlePointerLeave}
            style={{ cursor: `url(${selectionCursor}) ${CURSOR_OFFSET}, auto` }}
        >
            <PreviewAnnotations previewAnnotations={previewAnnotations} image={image} />
        </SvgToolCanvas>
    );
};
