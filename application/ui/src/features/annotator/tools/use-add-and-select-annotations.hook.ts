// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback } from 'react';

import type { Label } from '../../../constants/shared-types';
import { useAnnotationActions } from '../../../shared/annotator/annotation-actions-provider.component';
import { EMPTY_LABEL_ID } from '../../../shared/annotator/labels';
import { useSelectedAnnotations } from '../../../shared/annotator/select-annotation-provider.component';
import type { Shape } from '../../../shared/types';

export const useAddAndSelectAnnotations = () => {
    const { addAnnotations, annotations, deleteAnnotations } = useAnnotationActions();
    const { setSelectedAnnotations } = useSelectedAnnotations();

    const addAndSelectAnnotations = useCallback(
        (shapes: Shape[], labels: Label[]): string[] => {
            // If there is an annotation with empty label and we are trying to add new annotations,
            // delete annotation with empty label first
            if (annotations.some((annotation) => annotation.labels.some((label) => label.id === EMPTY_LABEL_ID))) {
                deleteAnnotations(annotations.map(({ id }) => id));
            }

            const newIds = addAnnotations(shapes, labels);
            setSelectedAnnotations(new Set(newIds));

            return newIds;
        },
        [addAnnotations, setSelectedAnnotations, annotations, deleteAnnotations]
    );

    return { addAndSelectAnnotations };
};
