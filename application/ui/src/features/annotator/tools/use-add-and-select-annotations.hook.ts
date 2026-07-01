// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback } from 'react';

import { isEmpty } from 'lodash-es';

import type { Label } from '../../../constants/shared-types';
import { useAnnotationActions } from '../../../shared/annotator/annotation-actions-provider.component';
import { useSelectedAnnotations } from '../../../shared/annotator/select-annotation-provider.component';
import type { Shape } from '../../../shared/types';

export const useAddAndSelectAnnotations = () => {
    const { addAnnotations, annotations, deleteAnnotations } = useAnnotationActions();
    const { setSelectedAnnotations } = useSelectedAnnotations();

    const addAndSelectAnnotations = useCallback(
        (shapes: Shape[], labels: Label[]): string[] => {
            // If there is a global annotation, and we are trying to add new annotations,
            // delete that global annotation first
            const globalAnnotations = annotations.filter((annotation) => annotation.shape.type === 'full_image');

            if (!isEmpty(globalAnnotations)) {
                deleteAnnotations(globalAnnotations.map(({ id }) => id));
            }

            const newIds = addAnnotations(
                shapes,
                labels.map(({ id }) => ({ id }))
            );
            setSelectedAnnotations(new Set(newIds));

            return newIds;
        },
        [addAnnotations, setSelectedAnnotations, annotations, deleteAnnotations]
    );

    return { addAndSelectAnnotations };
};
