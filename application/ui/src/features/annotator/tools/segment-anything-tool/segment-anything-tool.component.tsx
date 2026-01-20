// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, useRef, useState } from 'react';

import { clampPointBetweenImage } from '@geti/smart-tools/utils';

import { useZoom } from '../../../../components/zoom/zoom.provider';
import { Label } from '../../../../constants/shared-types';
import { useAnnotationActions } from '../../../../shared/annotator/annotation-actions-provider.component';
import { useAnnotator } from '../../../../shared/annotator/annotator-provider.component';
import type { Annotation, RegionOfInterest, Shape } from '../../../../shared/types';
import { AnnotationShape } from '../../annotations/annotation-shape.component';
import { MaskAnnotations } from '../../annotations/mask-annotations.component';
import { SvgToolCanvas } from '../svg-tool-canvas.component';
import { getRelativePoint, removeOffLimitPoints } from '../utils';
import { SAMLoading } from './sam-loading.component';
import { useSegmentAnythingModel } from './use-segment-anything.hook';
import { useSingleStackFn } from './use-single-stack-fn.hook';

import classes from './segment-anything.module.scss';

interface PreviewAnnotationsProps {
    previewAnnotations: Annotation[];
    image: Pick<RegionOfInterest, 'width' | 'height'>;
}

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
                    className={classes.animateStroke}
                >
                    <AnnotationShape annotation={annotation} />
                </g>
            ))}
        </MaskAnnotations>
    );
};

export const SegmentAnythingTool = () => {
    const [previewShapes, setPreviewShapes] = useState<Shape[]>([]);
    const [acceptedShapes, setAcceptedShapes] = useState<Shape[] | null>(null);
    const ref = useRef<SVGSVGElement>(null);

    const zoom = useZoom();
    const { roi, image, selectedLabel } = useAnnotator();
    const { addAnnotations } = useAnnotationActions();
    const { isLoading, decodingQueryFn } = useSegmentAnythingModel();
    const throttledDecodingQueryFn = useSingleStackFn(decodingQueryFn);

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

        throttledDecodingQueryFn([{ ...point, positive: true }])
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
        addAnnotations(shapes, [label]);
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

    const previewAnnotations = (acceptedShapes ?? previewShapes).map((shape, idx): Annotation => {
        return {
            shape,
            // During preview mode (while hovering), display the annotation without label color
            // to provide an unobscured view of the underlying image before finalizing placement.
            labels: [],
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
            onPointerLeave={() => setPreviewShapes([])}
            style={{ cursor: `url("/icons/selection.svg") 8 8, auto` }}
        >
            <PreviewAnnotations previewAnnotations={previewAnnotations} image={image} />
        </SvgToolCanvas>
    );
};
