// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Annotation as AnnotationType } from '../../../shared/types';
import { AnnotationContext } from './annotation-context';
import { AnnotationShapeRenderer } from './annotation-shape-renderer.component';
import { EditableAnnotation } from './editable-annotation.component';
import { SelectableAnnotation } from './selectable-annotation.component';

interface AnnotationProps {
    annotation: AnnotationType;
}
export const Annotation = ({ annotation }: AnnotationProps) => {
    return (
        <AnnotationContext.Provider value={annotation}>
            <SelectableAnnotation>
                <EditableAnnotation>
                    <AnnotationShapeRenderer annotation={annotation} />
                </EditableAnnotation>
            </SelectableAnnotation>
        </AnnotationContext.Provider>
    );
};
