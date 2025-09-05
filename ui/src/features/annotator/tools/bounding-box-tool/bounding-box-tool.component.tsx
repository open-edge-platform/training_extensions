// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Annotation, Point, RegionOfInterest } from '../../types';
import { ResizeAnchor } from './resize-anchor.component';
import { TranslateShape } from './translate-shape.component';
import { getBoundingBoxInRoi, getBoundingBoxResizePoints, getClampedBoundingBox } from './utils';

import classes from './bounding-box-tool.module.scss';

interface EditBoundingBoxProps {
    annotation: Annotation & { shape: { shapeType: 'rect' } };
    roi: RegionOfInterest;
    image: ImageData;
    zoom: number;
    updateAnnotation: (annotation: Annotation) => void;
}

const ANCHOR_SIZE = 8;

export const EditBoundingBox = ({ annotation, roi, image, zoom, updateAnnotation }: EditBoundingBoxProps) => {
    const [shape, setShape] = useState(annotation.shape);

    const onComplete = () => {
        updateAnnotation({ ...annotation, shape });
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
            <svg
                width={image.width}
                height={image.height}
                className={classes.disabledLayer}
                id={`translate-bounding-box-${annotation.id}`}
            >
                <TranslateShape
                    zoom={zoom}
                    annotation={{ ...annotation, shape }}
                    translateShape={translate}
                    onComplete={onComplete}
                />
            </svg>

            <svg
                width={image.width}
                height={image.height}
                className={classes.disabledLayer}
                aria-label={`Edit bounding box points ${annotation.id}`}
                id={`edit-bounding-box-points-${annotation.id}`}
            >
                <g style={{ pointerEvents: 'auto' }}>
                    {anchorPoints.map((anchor) => {
                        return <ResizeAnchor key={anchor.label} zoom={zoom} onComplete={onComplete} {...anchor} />;
                    })}
                </g>
            </svg>
        </>
    );
};
