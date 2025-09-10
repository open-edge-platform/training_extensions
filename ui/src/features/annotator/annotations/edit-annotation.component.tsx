// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { useSelectedAnnotations } from '../select-annotation-provider.component';
import { useAnnotation } from './annotation.component';

interface EditAnnotationProps {
    children: ReactNode;
}

export const EditAnnotation = ({ children }: EditAnnotationProps) => {
    const annotation = useAnnotation();
    const { selectedAnnotations } = useSelectedAnnotations();

    // Don't render the annotation if it's currently selected (being edited)
    const isSelected = selectedAnnotations?.has(annotation.id);

    if (isSelected) {
        return null;
    }

    return <>{children}</>;
};
