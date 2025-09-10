// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, useContext } from 'react';

import { Annotation as AnnotationInterface } from '../types';
import { AnnotationShape } from './annotation-shape.component';
import { EditAnnotation } from './edit-annotation.component';
import { SelectableAnnotation } from './selectable-annotation.component';

const AnnotationContext = createContext<AnnotationInterface | null>(null);

export const useAnnotation = () => {
    const ctx = useContext(AnnotationContext);

    return ctx!;
};

interface AnnotationProps {
    annotation: AnnotationInterface;
}
export const Annotation = ({ annotation }: AnnotationProps) => {
    return (
        <AnnotationContext.Provider value={annotation}>
            <EditAnnotation>
                <SelectableAnnotation>
                    <AnnotationShape annotation={annotation} />
                </SelectableAnnotation>
            </EditAnnotation>
        </AnnotationContext.Provider>
    );
};
