// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useSecondaryToolbarState } from 'src/features/dataset/media-preview/secondary-toolbar/use-secondary-toolbar-state.hook';

import { Annotation, Point, Polygon } from '../types';
import { AnnotationLabels } from './annotation-labels.component';

type AnnotationShapeProps = {
    annotation: Annotation;
};

const getFormattedPoints = (points: Point[]): string => {
    return points.map(({ x, y }) => `${x},${y}`).join(' ');
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

    // For polygon, find the top-left point to use as the group's origin
    const polygonPoints = (shape as Polygon).points;
    const minX = Math.min(...polygonPoints.map((point) => point.x));
    const minY = Math.min(...polygonPoints.map((point) => point.y));

    // Adjust polygon points to be relative to the group's origin
    const relativePoints = polygonPoints.map((point) => ({ x: point.x - minX, y: point.y - minY }));

    return (
        <g transform={`translate(${minX}, ${minY})`}>
            <polygon aria-label='annotation polygon' points={getFormattedPoints(relativePoints)} fill={color} />
            <AnnotationLabels labels={labels} onRemove={removeLabels} />
        </g>
    );
};
