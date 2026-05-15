// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useAnnotationVisibility } from './annotation-visibility-provider.component';
import { useSelectedAnnotations } from './select-annotation-provider.component';

export const useEditableAnnotationState = () => {
    const { isVisible } = useAnnotationVisibility();
    const { selectedAnnotations } = useSelectedAnnotations();

    const isSingleEditableSelection = isVisible && selectedAnnotations.size === 1;

    const isAnnotationEditable = (annotationId: string) => {
        return isSingleEditableSelection && selectedAnnotations.has(annotationId);
    };

    return { isSingleEditableSelection, isAnnotationEditable };
};
