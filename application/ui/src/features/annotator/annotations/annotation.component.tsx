// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useSelectedAnnotations } from '../../../shared/annotator/select-annotation-provider.component';
import type { Annotation as AnnotationType } from '../../../shared/types';
import { drawingStyles } from '../tools/polygon-tool/utils';
import { AnnotationContext } from './annotation-context';
import { AnnotationShapeRenderer } from './annotation-shape-renderer.component';

interface AnnotationProps {
    annotation: AnnotationType;
}
export const Annotation = ({ annotation }: AnnotationProps) => {
    const { selectedAnnotations } = useSelectedAnnotations();
    const isSelected = selectedAnnotations.has(annotation.id);

    if (isSelected) {
        return null;
    }

    return (
        <AnnotationContext.Provider value={annotation}>
            <g style={drawingStyles(annotation.labels?.[0] ?? null)}>
                <AnnotationShapeRenderer annotation={annotation} />
            </g>
        </AnnotationContext.Provider>
    );
};
