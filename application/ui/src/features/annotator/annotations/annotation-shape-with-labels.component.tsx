// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Key } from 'react';

import polylabel from 'polylabel';

import type { Label } from '../../../constants/shared-types';
import { useAnnotationActions } from '../../../shared/annotator/annotation-actions-provider.component';
import { useAnnotationVisibility } from '../../../shared/annotator/annotation-visibility-provider.component';
import { EMPTY_LABEL_ID } from '../../../shared/annotator/labels';
import type { Annotation } from '../../../shared/types';
import { useAnnotatorLabels } from '../annotator-labels-provider.component';
import { AnnotationLabels } from './annotation-labels/annotation-labels.component';
import { AnnotationShape } from './annotation-shape/annotation-shape.component';

type AnnotationShapeProps = {
    annotation: Annotation;
};

export const AnnotationShapeWithLabels = ({ annotation }: AnnotationShapeProps) => {
    const { isVisible } = useAnnotationVisibility();
    const { updateAnnotations, deleteAnnotations, isReadOnlyMode } = useAnnotationActions();
    const { selectedLabelId, setSelectedLabelId } = useAnnotatorLabels();

    const { shape, labels } = annotation;

    const removeLabels = (labelId: Key | null) => {
        if (isReadOnlyMode) {
            return;
        }

        if (labelId === EMPTY_LABEL_ID && selectedLabelId === EMPTY_LABEL_ID) {
            setSelectedLabelId(null);
        }

        const updatedLabels = annotation.labels.filter((label) => label.id !== labelId) as Label[];
        const hasNoLabels = updatedLabels.length === 0;

        if (hasNoLabels) {
            // An annotation without labels is invalid: remove it so it cannot be submitted.
            deleteAnnotations([annotation.id]);
        } else {
            updateAnnotations([{ ...annotation, labels: updatedLabels }]);
        }
    };

    if (shape.type === 'full_image') {
        return (
            <g display={isVisible ? 'block' : 'none'}>
                <AnnotationShape annotation={annotation} />
                <AnnotationLabels labels={labels} onRemove={removeLabels} isRemovable={!isReadOnlyMode} />
            </g>
        );
    }

    if (shape.type === 'rectangle') {
        return (
            <g transform={`translate(${shape.x}, ${shape.y})`} display={isVisible ? 'block' : 'none'}>
                <AnnotationShape annotation={{ ...annotation, shape: { ...shape, x: 0, y: 0 } }} />
                <AnnotationLabels labels={labels} onRemove={removeLabels} isRemovable={!isReadOnlyMode} />
            </g>
        );
    }

    const polygonCoords = [shape.points.map((point) => [point.x, point.y])];
    const [labelX, labelY] = polylabel(polygonCoords);

    return (
        <g transform={`translate(${labelX}, ${labelY})`} display={isVisible ? 'block' : 'none'}>
            <g transform={`translate(${-labelX}, ${-labelY})`}>
                <AnnotationShape annotation={annotation} />
            </g>
            <AnnotationLabels labels={labels} onRemove={removeLabels} useBottomCorners isRemovable={!isReadOnlyMode} />
        </g>
    );
};
