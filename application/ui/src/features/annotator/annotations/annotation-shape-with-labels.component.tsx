// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import polylabel from 'polylabel';
import { useSecondaryToolbarState } from 'src/features/dataset/media-preview/secondary-toolbar/use-secondary-toolbar-state.hook';

import { Annotation, Polygon } from '../types';
import { AnnotationLabels } from './annotation-labels.component';
import { getFormattedPoints } from './utils';

type AnnotationShapeProps = {
    annotation: Annotation;
};
export const AnnotationShapeWithLabels = ({ annotation }: AnnotationShapeProps) => {
    const { shape, labels } = annotation;
    const color = labels.length ? labels[0].color : '--annotation-fill';
    const { removeLabels } = useSecondaryToolbarState();

    if (shape.type === 'rectangle') {
        return (
            <g transform={`translate(${shape.x}, ${shape.y})`}>
                <rect aria-label='annotation rect' x={0} y={0} width={shape.width} height={shape.height} fill={color} />
                <AnnotationLabels labels={labels} onRemove={removeLabels} />
            </g>
        );
    }

    const polygonPoints = (shape as Polygon).points;
    const polygonCoords = [polygonPoints.map((point) => [point.x, point.y])];
    const [labelX, labelY] = polylabel(polygonCoords);

    return (
        <g transform={`translate(${labelX}, ${labelY})`}>
            <polygon
                aria-label='annotation polygon'
                points={getFormattedPoints(polygonPoints)}
                fill={color}
                transform={`translate(${-labelX}, ${-labelY})`}
            />
            <AnnotationLabels labels={labels} onRemove={removeLabels} />
        </g>
    );
};
