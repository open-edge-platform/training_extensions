// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Annotation as AnnotationType } from '../types';
import { AnnotationContext } from './annotation-context';
import { AnnotationShapeWithLabels } from './annotation-shape-with-labels.component';
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
                    <AnnotationShapeWithLabels annotation={annotation} />
                </EditableAnnotation>
            </SelectableAnnotation>
        </AnnotationContext.Provider>
    );
};
