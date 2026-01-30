// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback } from 'react';

import type { Label } from '../../../constants/shared-types';
import { useAnnotationActions } from '../../../shared/annotator/annotation-actions-provider.component';
import { useSelectedAnnotations } from '../../../shared/annotator/select-annotation-provider.component';
import type { Shape } from '../../../shared/types';

export const useAddAndSelectAnnotations = () => {
    const { addAnnotations } = useAnnotationActions();
    const { setSelectedAnnotations } = useSelectedAnnotations();

    const addAndSelectAnnotations = useCallback(
        (shapes: Shape[], labels: Label[]): string[] => {
            const newIds = addAnnotations(shapes, labels);
            setSelectedAnnotations(new Set(newIds));
            return newIds;
        },
        [addAnnotations, setSelectedAnnotations]
    );

    return { addAndSelectAnnotations };
};
