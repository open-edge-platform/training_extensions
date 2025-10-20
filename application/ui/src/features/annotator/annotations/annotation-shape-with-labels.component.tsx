// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import polylabel from 'polylabel';
import { useSecondaryToolbarState } from 'src/features/dataset/media-preview/secondary-toolbar/use-secondary-toolbar-state.hook';

import { useAnnotationVisibility } from '../annotation-visibility-provider.component';
import { Annotation, Polygon } from '../types';
import { AnnotationLabels } from './annotation-labels.component';
import { AnnotationShape } from './annotation-shape.component';

type AnnotationShapeProps = {
    annotation: Annotation;
};

export const AnnotationShapeWithLabels = ({ annotation }: AnnotationShapeProps) => {
    const { shape, labels } = annotation;
    const { removeLabels } = useSecondaryToolbarState();
    const { isVisible } = useAnnotationVisibility();

    if (shape.type === 'rectangle') {
        return (
            <g transform={`translate(${shape.x}, ${shape.y})`} display={isVisible ? 'block' : 'none'}>
                <AnnotationShape annotation={{ ...annotation, shape: { ...shape, x: 0, y: 0 } }} />
                <AnnotationLabels labels={labels} onRemove={removeLabels} />
            </g>
        );
    }

    const polygonPoints = (shape as Polygon).points;
    const polygonCoords = [polygonPoints.map((point) => [point.x, point.y])];
    const [labelX, labelY] = polylabel(polygonCoords);

    return (
        <g transform={`translate(${labelX}, ${labelY})`} display={isVisible ? 'block' : 'none'}>
            <g transform={`translate(${-labelX}, ${-labelY})`}>
                <AnnotationShape annotation={annotation} />
            </g>
            <AnnotationLabels labels={labels} onRemove={removeLabels} />
        </g>
    );
};
