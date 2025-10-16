// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import polylabel from 'polylabel';
import { useSecondaryToolbarState } from 'src/features/dataset/media-preview/secondary-toolbar/use-secondary-toolbar-state.hook';

import { Annotation, Polygon } from '../types';
import { AnnotationLabels } from './annotation-labels.component';
import { AnnotationShape } from './annotation-shape.component';

type AnnotationShapeProps = {
    annotation: Annotation;
};

export const AnnotationShapeWithLabels = ({ annotation }: AnnotationShapeProps) => {
    const { shape, labels } = annotation;
    const { removeLabels } = useSecondaryToolbarState();

    if (shape.type === 'rectangle') {
        return (
            <g transform={`translate(${shape.x}, ${shape.y})`}>
                <AnnotationShape annotation={{ ...annotation, shape: { ...shape, x: 0, y: 0 } }} />
                <AnnotationLabels labels={labels} onRemove={removeLabels} />
            </g>
        );
    }

    // For polygon, use polylabel to find the optimal label position
    const polygonPoints = (shape as Polygon).points;
    const polygonCoords = [polygonPoints.map((point) => [point.x, point.y])];
    const [labelX, labelY] = polylabel(polygonCoords, 1.0);

    return (
        <g transform={`translate(${labelX}, ${labelY})`}>
            <g transform={`translate(${-labelX}, ${-labelY})`}>
                <AnnotationShape annotation={annotation} />
            </g>
            <AnnotationLabels labels={labels} onRemove={removeLabels} />
        </g>
    );
};
