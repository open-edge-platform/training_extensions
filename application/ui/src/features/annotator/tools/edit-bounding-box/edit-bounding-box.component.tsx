// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { useAnnotationActions } from '../../../../shared/annotator/annotation-actions-provider.component';
import { useAnnotator } from '../../../../shared/annotator/annotator-provider.component';
import { AnnotationShapeWithLabels } from '../../annotations/annotation-shape-with-labels.component';
import type { Annotation, Point } from '../../types';
import { getBoundingBoxInRoi, getBoundingBoxResizePoints, getClampedBoundingBox } from '../utils';
import { ANCHOR_SIZE, ResizeAnchor } from './resize-anchor.component';
import { TranslateShape } from './translate-shape.component';

interface EditBoundingBoxProps {
    annotation: Annotation & { shape: { type: 'rectangle' } };
    zoom: number;
}

export const EditBoundingBox = ({ annotation, zoom }: EditBoundingBoxProps) => {
    const [shape, setShape] = useState(annotation.shape);
    const { mediaItem } = useAnnotator();
    const { updateAnnotations } = useAnnotationActions();

    const roi = { x: 0, y: 0, width: mediaItem.width, height: mediaItem.height };

    const onComplete = () => {
        updateAnnotations([{ ...annotation, shape }]);
    };

    const translate = (point: Point) => {
        const newBoundingBox = getClampedBoundingBox(point, shape, roi);

        setShape({ ...shape, ...newBoundingBox });
    };

    const anchorPoints = getBoundingBoxResizePoints({
        gap: (2 * ANCHOR_SIZE) / zoom,
        boundingBox: shape,
        onResized: (boundingBox) => {
            setShape({ ...shape, ...getBoundingBoxInRoi(boundingBox, roi) });
        },
    });

    return (
        <>
            <TranslateShape
                zoom={zoom}
                annotation={{ ...annotation, shape }}
                translateShape={translate}
                onComplete={onComplete}
            >
                <AnnotationShapeWithLabels annotation={{ ...annotation, shape }} />
            </TranslateShape>

            <g
                style={{ pointerEvents: 'auto' }}
                aria-label={`Edit bounding box points ${annotation.id}`}
                id={`edit-bounding-box-points-${annotation.id}`}
            >
                {anchorPoints.map((anchor) => {
                    return <ResizeAnchor key={anchor.label} zoom={zoom} onComplete={onComplete} {...anchor} />;
                })}
            </g>
        </>
    );
};
